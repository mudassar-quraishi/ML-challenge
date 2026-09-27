import os
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .config import PipelineConfig
from .features import build_feature_matrix
from .normalization import normalize_dataframe
from .make_submission import format_id_list


def run_batch_inference(
    model: LGBMClassifier,
    candidates_map: Dict[str, List[str]],
    s1_dict: Dict[str, dict],
    target_dict: Dict[str, dict],
    threshold: float,
    batch_size: int = 1000
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Score candidates in batches and produce final candidate and matched sets."""
    print(f"[Predict] Scoring candidates for {len(candidates_map)} S1 entities at threshold {threshold:.2f}...")
    
    final_candidates: Dict[str, List[str]] = {}
    final_matches: Dict[str, List[str]] = {}
    
    all_s1_ids = list(candidates_map.keys())
    
    for i in range(0, len(all_s1_ids), batch_size):
        chunk_s1 = all_s1_ids[i:i + batch_size]
        chunk_pairs: List[Tuple[str, str]] = []
        pair_to_s1: List[str] = []
        pair_to_cid: List[str] = []
        
        for s1_id in chunk_s1:
            cands = candidates_map.get(s1_id, [])
            final_candidates[s1_id] = list(cands)
            final_matches[s1_id] = []
            for cid in cands:
                chunk_pairs.append((s1_id, cid))
                pair_to_s1.append(s1_id)
                pair_to_cid.append(cid)
                
        if not chunk_pairs:
            continue
            
        X_chunk = build_feature_matrix(chunk_pairs, s1_dict, target_dict)
        probs = model.predict_proba(X_chunk)[:, 1]
        
        for s1_id, cid, prob in zip(pair_to_s1, pair_to_cid, probs):
            if prob >= threshold:
                final_matches[s1_id].append(cid)
                
    return final_candidates, final_matches


def stream_chunked_test_inference(
    model: LGBMClassifier,
    threshold: float,
    test_blocker,
    s1_path: str,
    output_dir: str,
    chunk_size: int = 5000,
    max_s1: Optional[int] = None
) -> Tuple[str, str, int, int]:
    """Stream test inference in memory-safe chunks, appending directly to submission TSVs.
    
    Fully resumable: skips already-processed S1 IDs if output files already exist.
    Guarantees:
    - Exactly 1 row per test S1 entity
    - Matches strictly subset of candidates
    - Only valid S2- and S3- prefixes
    - Constant RAM footprint regardless of test dataset size (1.7M+ entities)
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # Check resumability
    existing_ids: Set[str] = set()
    if os.path.exists(matching_path) and os.path.getsize(matching_path) > 0:
        with open(matching_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.split("\t", 1)
                if parts and parts[0]:
                    existing_ids.add(parts[0].strip())
        print(f"[StreamingInference] Resuming run: found {len(existing_ids):,} existing completed S1 records.")
        write_header = False
    else:
        write_header = True

    f_match = open(matching_path, "a" if not write_header else "w", encoding="utf-8")
    f_cand = open(candidate_path, "a" if not write_header else "w", encoding="utf-8")
    
    if write_header:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.flush()
        f_cand.flush()

    print(f"[StreamingInference] Beginning streaming inference on {s1_path} (chunk_size={chunk_size})...")
    total_processed_s1 = len(existing_ids)
    total_candidates_count = 0
    total_matches_count = 0
    
    reader = pd.read_csv(s1_path, sep="\t", chunksize=chunk_size, dtype=str)
    
    try:
        for chunk_idx, raw_chunk in enumerate(reader):
            if max_s1 and total_processed_s1 >= max_s1:
                print(f"[StreamingInference] Reached smoke test limit of {max_s1} S1 entities.")
                break
                
            # Filter out any already-completed entities (resumability)
            if existing_ids:
                raw_chunk = raw_chunk[~raw_chunk["entity_id"].isin(existing_ids)]
                if raw_chunk.empty:
                    continue
                    
            if max_s1 and (total_processed_s1 + len(raw_chunk)) > max_s1:
                raw_chunk = raw_chunk.iloc[:max_s1 - total_processed_s1]
                
            chunk_df = normalize_dataframe(raw_chunk)
            
            # Generate candidates for chunk
            cands_map = test_blocker.generate_candidates(chunk_df)
            
            # Metadata lookup for pairwise features
            chunk_s1_dict = {
                row["entity_id"]: row.to_dict() for _, row in chunk_df.iterrows()
            }
            
            # Prepare pairs
            chunk_pairs: List[Tuple[str, str]] = []
            chunk_matches: Dict[str, List[str]] = {s1_id: [] for s1_id in chunk_df["entity_id"]}
            chunk_cands: Dict[str, List[str]] = {s1_id: cands_map.get(s1_id, []) for s1_id in chunk_df["entity_id"]}
            
            for s1_id, cands in chunk_cands.items():
                for cid in cands:
                    chunk_pairs.append((s1_id, cid))
                    
            if chunk_pairs:
                X_chunk = build_feature_matrix(chunk_pairs, chunk_s1_dict, test_blocker.target_records)
                probs = model.predict_proba(X_chunk)[:, 1]
                for (s1_id, cid), prob in zip(chunk_pairs, probs):
                    if prob >= threshold:
                        chunk_matches[s1_id].append(cid)
                        
            # Write chunk rows directly to TSVs
            for s1_id in chunk_df["entity_id"].values:
                cands = chunk_cands.get(s1_id, [])
                matches = chunk_matches.get(s1_id, [])
                
                valid_cands = [c for c in cands if c.startswith(("S2-", "S3-"))]
                valid_cands_set = set(valid_cands)
                valid_matches = [m for m in matches if m in valid_cands_set]
                
                cand_str = format_id_list(valid_cands)
                match_str = format_id_list(valid_matches)
                
                f_match.write(f"{s1_id}\t{match_str}\n")
                f_cand.write(f"{s1_id}\t{cand_str}\n")
                
                total_candidates_count += len(valid_cands)
                total_matches_count += len(valid_matches)
                total_processed_s1 += 1
                existing_ids.add(s1_id)
                
            f_match.flush()
            f_cand.flush()
            
            if (chunk_idx + 1) % 5 == 0 or total_processed_s1 >= (max_s1 or float("inf")):
                print(f"[StreamingInference] Progress: {total_processed_s1:,} S1 entities processed | "
                      f"{total_candidates_count:,} candidates | {total_matches_count:,} matches.")
                      
    finally:
        f_match.close()
        f_cand.close()
        
    print(f"[StreamingInference] Streaming complete. Total S1 entities written: {total_processed_s1:,}")
    return matching_path, candidate_path, total_processed_s1, total_candidates_count

"""Inference and candidate scoring module for Business Entity Resolution.

Scores candidate pairs in memory-efficient batches and applies calibrated decision threshold.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .config import PipelineConfig
from .features import build_feature_matrix


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

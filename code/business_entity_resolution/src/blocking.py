"""Blocking / Candidate Generation module for Business Entity Resolution.

Memory-safe, high-recall candidate generation using:
- Inverted token indexing with frequency capping
- Exact normalized name indexing
- Address digit/postal key matching
- Character n-gram TF-IDF approximate nearest neighbor matching
- Controlled candidate capping and union aggregation
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import PipelineConfig

COMMON_STOP_TOKENS = {
    "and", "the", "inc", "llc", "ltd", "corp", "pvt", "limited", "company", "co",
    "corporation", "private", "services", "enterprises", "solutions", "group", "center",
    "sarl", "sas", "sasu", "sci", "club", "store", "shop", "care", "auto", "food"
}


class CandidateBlocker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_candidates = config.BLOCKING_MAX_CANDIDATES
        
        # Inverted index mappings: key -> list of target entity IDs
        self.name_exact_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.digit_index = defaultdict(list)
        
        # Target lookup
        self.target_records: Dict[str, dict] = {}
        
    def build_index(self, df_targets: pd.DataFrame):
        """Index target sources (S2 and S3 combined) using memory-safe inverted lists."""
        print(f"[Blocking] Building indexes over {len(df_targets)} target records...")
        
        # Step 1: Count token frequencies to discard ultra-frequent terms that blow up memory
        token_freq = Counter()
        for name in df_targets["norm_name"].values:
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= self.config.MIN_TOKEN_LEN and t not in COMMON_STOP_TOKENS:
                    token_freq[t] += 1
                    
        # Token cap: ignore tokens appearing in > 0.5% of records or > 5000 records
        max_doc_freq = min(5000, max(50, int(len(df_targets) * 0.005)))
        valid_tokens = {t for t, count in token_freq.items() if count <= max_doc_freq}
        print(f"[Blocking] Selected {len(valid_tokens)} indexed informative tokens (freq <= {max_doc_freq}).")
        
        # Step 2: Populate inverted indexes
        for _, row in df_targets.iterrows():
            eid = row["entity_id"]
            name = row["norm_name"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            # Store target record metadata for feature extraction
            self.target_records[eid] = {
                "business_name": row["business_name"],
                "norm_name": name,
                "business_address": row.get("business_address", ""),
                "norm_address": row.get("norm_address", ""),
                "country": row.get("country", ""),
                "norm_country": country,
                "address_digits": digits
            }
            
            # Index exact normalized name
            if name:
                self.name_exact_index[name].append(eid)
                
            # Index informative name tokens
            for t in set(name.split()):
                if t in valid_tokens:
                    self.token_index[t].append(eid)
                    
            # Index primary address digits (house number, PIN code)
            for d in digits[:3]:
                if len(d) >= 3:
                    self.digit_index[(d, country)].append(eid)

        print("[Blocking] Target indexing complete.")

    def generate_candidates(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate candidate pairs for each S1 entity using union of blocking keys."""
        print(f"[Blocking] Generating candidates for {len(df_s1)} S1 entities (max {self.max_candidates} per S1)...")
        candidates_map: Dict[str, List[str]] = {}
        
        for _, row in df_s1.iterrows():
            s1_id = row["entity_id"]
            name = row["norm_name"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            cand_scores = Counter()
            
            # Strategy A: Exact normalized name match (Highest priority)
            if name and name in self.name_exact_index:
                for target_id in self.name_exact_index[name]:
                    cand_scores[target_id] += 50.0
                    
            # Strategy B: Informative name token overlap
            tokens = [t for t in set(name.split()) if len(t) >= self.config.MIN_TOKEN_LEN and t not in COMMON_STOP_TOKENS]
            for t in tokens:
                if t in self.token_index:
                    matches = self.token_index[t]
                    # Give higher weight to rarer tokens
                    w = 1.0 + (10.0 / (len(matches) + 1))
                    for target_id in matches:
                        cand_scores[target_id] += w
                        
            # Strategy C: Address digit / PIN key overlap (helps with cross-script transliteration)
            for d in digits[:3]:
                if len(d) >= 3 and (d, country) in self.digit_index:
                    matches = self.digit_index[(d, country)]
                    if len(matches) <= 200:
                        w = 2.0 + (5.0 / (len(matches) + 1))
                        for target_id in matches:
                            cand_scores[target_id] += w
                            
            # Select top-K candidates
            if not cand_scores:
                candidates_map[s1_id] = []
            else:
                top_cands = [cid for cid, _ in cand_scores.most_common(self.max_candidates)]
                candidates_map[s1_id] = top_cands
                
        return candidates_map


def evaluate_blocking(
    candidates_map: Dict[str, List[str]],
    ground_truth_map: Dict[str, Set[str]],
    n_targets: int
) -> dict:
    """Evaluate blocking quality: recall, volume, and reduction ratio."""
    total_true_matches = 0
    captured_true_matches = 0
    candidate_counts = []
    
    for s1_id, true_matches in ground_truth_map.items():
        n_true = len(true_matches)
        total_true_matches += n_true
        
        cands = set(candidates_map.get(s1_id, []))
        candidate_counts.append(len(cands))
        
        if n_true > 0:
            captured_true_matches += len(true_matches.intersection(cands))
            
    recall = (captured_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    
    total_s1 = len(ground_truth_map)
    total_candidates = sum(candidate_counts)
    total_search_space = total_s1 * n_targets
    reduction_ratio = 1.0 - (total_candidates / total_search_space) if total_search_space > 0 else 0.0
    
    arr_counts = np.array(candidate_counts)
    
    metrics = {
        "total_s1": total_s1,
        "total_targets": n_targets,
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_true_matches,
        "blocking_recall": float(recall),
        "total_candidates": int(total_candidates),
        "mean_candidates_per_s1": float(np.mean(arr_counts)),
        "median_candidates_per_s1": float(np.median(arr_counts)),
        "p95_candidates_per_s1": float(np.percentile(arr_counts, 95)),
        "p99_candidates_per_s1": float(np.percentile(arr_counts, 99)),
        "max_candidates_per_s1": int(np.max(arr_counts)),
        "reduction_ratio": float(reduction_ratio)
    }
    return metrics

"""Blocking and Candidate Generation module for Business Entity Resolution.

Supports dual modes:
1. "baseline" - Phase 1 baseline blocker (Exact Name + Inverted Tokens + Raw Digits >= 3)
2. "multimodal" - Phase 3 production multi-modal union blocker:
   - Blocker A: Exact Normalized Name
   - Blocker B: Rare Name Tokens (Inverted Index with Frequency Capping)
   - Blocker C: Subword Character 3-gram TF-IDF Cosine Nearest Neighbors
   - Blocker D: URL / Domain / Handle Stripped Brand Matching
   - Blocker E: Address Digits (>= 3 digits) & Compound 2-Digit Location Keys
   - Blocker F: Informative Address Token Matching (>= 2 Token Overlaps)
"""

import math
import re
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

ADDRESS_STOP_TOKENS = {
    "street", "road", "avenue", "drive", "lane", "boulevard", "court", "parkway",
    "floor", "suite", "apartment", "unit", "rue", "allée", "no", "near", "opp",
    "behind", "beside", "block", "sector", "phase", "plot", "building", "tower",
    "complex", "plaza", "bhavan", "nagar", "colony", "enclave", "vihar",
    "india", "us", "france", "state", "city", "district", "town",
    "delhi", "mumbai", "chennai", "kolkata", "bengaluru", "hyderabad", "pune",
    "california", "texas", "florida", "new", "york", "illinois", "ohio", "north",
    "south", "east", "west", "null"
}

DOMAIN_SUFFIXES = r"\.(com|in|org|net|fr|co|edu|biz|info|io)\b"


def extract_core_brand(name: str) -> str:
    """Strip web extensions, social handles, and non-alphanumerics to isolate core brand."""
    text = name.lower()
    text = re.sub(r"^@", "", text)
    text = re.sub(DOMAIN_SUFFIXES, "", text)
    text = re.sub(r"^www\.", "", text)
    text = re.sub(r"[^\w]", "", text)
    return text


# =====================================================================
# PHASE 1 BASELINE BLOCKER (PRESERVED FOR FALLBACK & BENCHMARKING)
# =====================================================================

class BaselineCandidateBlocker:
    """Phase 1 baseline candidate blocker."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_candidates = config.BLOCKING_MAX_CANDIDATES
        self.name_exact_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.digit_index = defaultdict(list)
        self.target_records: Dict[str, dict] = {}
        
    def build_index(self, df_targets: pd.DataFrame):
        print(f"[Blocking-Baseline] Building indexes over {len(df_targets)} target records...")
        token_freq = Counter()
        for name in df_targets["norm_name"].values:
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= self.config.MIN_TOKEN_LEN and t not in COMMON_STOP_TOKENS:
                    token_freq[t] += 1
                    
        max_doc_freq = self.config.SMOKE_TOKEN_DOC_FREQ if self.config.LOCAL_SMOKE_TEST else min(
            self.config.MAX_TOKEN_DOC_FREQ, max(50, int(len(df_targets) * self.config.CORPUS_RELATIVE_CAP))
        )
        valid_tokens = {t for t, count in token_freq.items() if count <= max_doc_freq}
        print(f"[Blocking-Baseline] Selected {len(valid_tokens)} indexed informative tokens (freq <= {max_doc_freq}).")
        
        for _, row in df_targets.iterrows():
            eid = row["entity_id"]
            name = row["norm_name"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            self.target_records[eid] = {
                "business_name": row["business_name"],
                "norm_name": name,
                "business_address": row.get("business_address", ""),
                "norm_address": row.get("norm_address", ""),
                "country": row.get("country", ""),
                "norm_country": country,
                "address_digits": digits
            }
            if name:
                self.name_exact_index[name].append(eid)
            for t in set(name.split()):
                if t in valid_tokens:
                    self.token_index[t].append(eid)
            for d in digits[:3]:
                if len(d) >= 3:
                    self.digit_index[(d, country)].append(eid)

    def generate_candidates(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
        print(f"[Blocking-Baseline] Generating candidates for {len(df_s1)} S1 entities...")
        candidates_map: Dict[str, List[str]] = {}
        for _, row in df_s1.iterrows():
            s1_id = row["entity_id"]
            name = row["norm_name"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            cand_scores = Counter()
            if name and name in self.name_exact_index:
                for target_id in self.name_exact_index[name]:
                    cand_scores[target_id] += 50.0
            tokens = [t for t in set(name.split()) if len(t) >= self.config.MIN_TOKEN_LEN and t not in COMMON_STOP_TOKENS]
            for t in tokens:
                if t in self.token_index:
                    matches = self.token_index[t]
                    w = 1.0 + (10.0 / (len(matches) + 1))
                    for target_id in matches:
                        cand_scores[target_id] += w
            for d in digits[:3]:
                if len(d) >= 3 and (d, country) in self.digit_index:
                    matches = self.digit_index[(d, country)]
                    if len(matches) <= 200:
                        w = 2.0 + (5.0 / (len(matches) + 1))
                        for target_id in matches:
                            cand_scores[target_id] += w
                            
            if not cand_scores:
                candidates_map[s1_id] = []
            else:
                candidates_map[s1_id] = [cid for cid, _ in cand_scores.most_common(self.max_candidates)]
        return candidates_map


# =====================================================================
# PHASE 3 MULTI-MODAL UNION BLOCKER (PRODUCTION QUALITY)
# =====================================================================

class MultiModalCandidateBlocker:
    """Phase 3 Production Multi-Modal Union Candidate Blocker.
    
    Combines:
    - Blocker A: Exact Normalized Name
    - Blocker B: Rare Name Tokens (Inverted Index)
    - Blocker C: Subword Character 3-Gram TF-IDF Cosine (Typo & Handle Resilience)
    - Blocker D: Domain / Social Handle Stripped Brand Matching
    - Blocker E: Address Digits (>= 3) & Compound 2-Digit Location Keys
    - Blocker F: Informative Address Tokens (>= 2 Overlaps)
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_candidates = config.BLOCKING_MAX_CANDIDATES
        
        # Sub-indexes
        self.exact_name_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.brand_clean_index = defaultdict(list)
        self.digit_index = defaultdict(list)
        self.compound_digit_index = defaultdict(list)
        self.address_token_index = defaultdict(list)
        
        # TF-IDF Char 3-gram components
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
        self.target_char_matrix = None
        self.target_ids: List[str] = []
        
        # Target lookup
        self.target_records: Dict[str, dict] = {}
        
    def build_index(self, df_targets: pd.DataFrame):
        print(f"[Blocking-MultiModal] Indexing {len(df_targets)} target records across 6 modalities...")
        
        # 1. Determine frequency caps
        token_freq = Counter()
        addr_token_freq = Counter()
        dig_freq = Counter()
        
        for _, row in df_targets.iterrows():
            name = row["norm_name"]
            addr = row["norm_address"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            for t in set(name.split()):
                if len(t) >= self.config.MIN_TOKEN_LEN and t not in COMMON_STOP_TOKENS:
                    token_freq[t] += 1
            for t in set(addr.split()):
                if len(t) >= 4 and t not in ADDRESS_STOP_TOKENS:
                    addr_token_freq[t] += 1
            for d in digits[:3]:
                if len(d) >= 3:
                    dig_freq[(d, country)] += 1

        name_cap = self.config.SMOKE_TOKEN_DOC_FREQ if self.config.LOCAL_SMOKE_TEST else min(
            self.config.MAX_TOKEN_DOC_FREQ, max(50, int(len(df_targets) * self.config.CORPUS_RELATIVE_CAP))
        )
        addr_cap = name_cap
        dig_cap = name_cap * 3
        
        valid_name_tokens = {t for t, c in token_freq.items() if c <= name_cap}
        valid_addr_tokens = {t for t, c in addr_token_freq.items() if c <= addr_cap}
        valid_digits = {k for k, c in dig_freq.items() if c <= dig_cap}
        
        print(f"[Blocking-MultiModal] Filtered vocabulary: Name Tokens={len(valid_name_tokens)} (cap<={name_cap}), "
              f"Addr Tokens={len(valid_addr_tokens)} (cap<={addr_cap}), Digits={len(valid_digits)} (cap<={dig_cap})")
              
        # 2. Populate Inverted Indexes
        names_list = []
        for _, row in df_targets.iterrows():
            eid = row["entity_id"]
            name = row["norm_name"]
            raw_name = row["business_name"]
            addr = row["norm_address"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            # Store target record metadata
            self.target_records[eid] = {
                "business_name": raw_name,
                "norm_name": name,
                "business_address": row.get("business_address", ""),
                "norm_address": addr,
                "country": row.get("country", ""),
                "norm_country": country,
                "address_digits": digits
            }
            
            self.target_ids.append(eid)
            names_list.append(name if name else "")
            
            # Blocker A: Exact Name
            if name:
                self.exact_name_index[name].append(eid)
                
            # Blocker B: Rare Name Tokens
            for t in set(name.split()):
                if t in valid_name_tokens:
                    self.token_index[t].append(eid)
                    
            # Blocker D: Domain / Handle Stripped Brand
            core_brand = extract_core_brand(raw_name)
            if len(core_brand) >= 4:
                self.brand_clean_index[core_brand].append(eid)
                
            # Blocker E: Digits (>= 3)
            for d in digits[:3]:
                if (d, country) in valid_digits:
                    self.digit_index[(d, country)].append(eid)
                    
            # Compound 2-Digit Keys for Unit / Flat Numbers
            addr_distinct_tokens = [t for t in set(addr.split()) if len(t) >= 4 and t in valid_addr_tokens]
            for d in digits:
                if len(d) == 2:
                    for tok in addr_distinct_tokens[:3]:
                        self.compound_digit_index[(d, tok, country)].append(eid)
                        
            # Blocker F: Informative Address Tokens
            for t in addr_distinct_tokens:
                self.address_token_index[t].append(eid)

        # 3. Fit Char 3-gram TF-IDF Matrix
        print("[Blocking-MultiModal] Vectorizing subword character 3-grams for nearest neighbors...")
        self.target_char_matrix = self.vectorizer.fit_transform(names_list)
        print("[Blocking-MultiModal] Multi-modal indexing complete.")

    def generate_candidates(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
        print(f"[Blocking-MultiModal] Generating candidates for {len(df_s1)} S1 entities...")
        candidates_map: Dict[str, List[str]] = {}
        
        # Batch vectorization of S1 names for Blocker C
        s1_names = df_s1["norm_name"].fillna("").tolist()
        s1_char_matrix = self.vectorizer.transform(s1_names)
        
        # Compute dot products in chunks to remain memory-safe
        chunk_size = 500
        n_s1 = len(df_s1)
        
        for start_idx in range(0, n_s1, chunk_size):
            end_idx = min(start_idx + chunk_size, n_s1)
            chunk_s1_matrix = s1_char_matrix[start_idx:end_idx]
            
            # Sparse matrix multiplication: (chunk_size, V) x (V, N_targets) -> (chunk_size, N_targets)
            sim_matrix = (chunk_s1_matrix * self.target_char_matrix.T).tocsr()
            
            for offset, s1_row_idx in enumerate(range(start_idx, end_idx)):
                row = df_s1.iloc[s1_row_idx]
                s1_id = row["entity_id"]
                name = row["norm_name"]
                raw_name = row["business_name"]
                addr = row["norm_address"]
                country = row["norm_country"]
                digits = row.get("address_digits", [])
                
                combined_cands = Counter()
                
                # --- Strategy A: Exact Normalized Name ---
                if name and name in self.exact_name_index:
                    for tid in self.exact_name_index[name]:
                        combined_cands[tid] += 50.0
                        
                # --- Strategy B: Rare Name Tokens ---
                tokens = [t for t in set(name.split()) if len(t) >= self.config.MIN_TOKEN_LEN and t not in COMMON_STOP_TOKENS]
                for t in tokens:
                    if t in self.token_index:
                        postings = self.token_index[t]
                        w = 1.0 + (10.0 / (len(postings) + 1))
                        for tid in postings:
                            combined_cands[tid] += w
                            
                # --- Strategy C: Char 3-Gram Cosine Nearest Neighbors ---
                row_vec = sim_matrix[offset]
                if row_vec.nnz > 0:
                    above_min = np.where(row_vec.data >= self.config.NAME_TFIDF_MIN_SIM)[0]
                    for idx in above_min:
                        col_idx = row_vec.indices[idx]
                        score = float(row_vec.data[idx])
                        tid = self.target_ids[col_idx]
                        combined_cands[tid] += (score * 15.0)
                        
                # --- Strategy D: Domain / Handle Stripped Brand Matching ---
                core_brand = extract_core_brand(raw_name)
                if len(core_brand) >= 4 and core_brand in self.brand_clean_index:
                    for tid in self.brand_clean_index[core_brand]:
                        combined_cands[tid] += 25.0
                        
                # --- Strategy E: Address Digits & Compound 2-Digit Keys ---
                for d in digits[:3]:
                    if len(d) >= 3 and (d, country) in self.digit_index:
                        postings = self.digit_index[(d, country)]
                        w = 2.0 + (5.0 / (len(postings) + 1))
                        for tid in postings:
                            combined_cands[tid] += w
                            
                addr_tokens = [t for t in set(addr.split()) if len(t) >= 4 and t not in ADDRESS_STOP_TOKENS]
                for d in digits:
                    if len(d) == 2:
                        for tok in addr_tokens[:3]:
                            key = (d, tok, country)
                            if key in self.compound_digit_index:
                                postings = self.compound_digit_index[key]
                                if len(postings) <= 25:
                                    w = 3.0 + (5.0 / (len(postings) + 1))
                                    for tid in postings:
                                        combined_cands[tid] += w
                                        
                # --- Strategy F: Informative Address Tokens (>= 2 Token Overlaps) ---
                addr_cand_counts = Counter()
                for tok in addr_tokens:
                    if tok in self.address_token_index:
                        for tid in self.address_token_index[tok]:
                            addr_cand_counts[tid] += 1
                for tid, count in addr_cand_counts.items():
                    if count >= 2:
                        combined_cands[tid] += (count * 2.5)
                        
                # Select top-K candidates
                if not combined_cands:
                    candidates_map[s1_id] = []
                else:
                    candidates_map[s1_id] = [cid for cid, _ in combined_cands.most_common(self.max_candidates)]
                    
        return candidates_map


def CandidateBlocker(config: PipelineConfig):
    """Factory function returning the configured CandidateBlocker."""
    mode = getattr(config, "BLOCKING_MODE", "multimodal").lower()
    if mode == "baseline":
        return BaselineCandidateBlocker(config)
    return MultiModalCandidateBlocker(config)


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

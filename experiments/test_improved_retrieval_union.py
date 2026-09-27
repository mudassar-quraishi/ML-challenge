"""Phase 4.3: Validate Integrated Improved Retrieval on 10,000 S1 Universe.

Integrates the validated retrieval enhancements:
1. Subword Character (2, 3)-gram TF-IDF
2. Low-collision digits (len >= 2, postings <= 30)
3. Rare locality address tokens (len >= 5, postings <= 20)

Measures:
- New raw retrieval recall (vs previous 97.57%)
- New Tiered 85 blocking recall (vs previous 94.96%)
- Unique matches recovered from the 848 previously missed true matches
- Candidate volume, P95, RAM, and runtime
"""

import os
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe, extract_address_digits
from src.blocking import extract_core_brand, COMMON_STOP_TOKENS, ADDRESS_STOP_TOKENS, evaluate_blocking

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def get_peak_mem_mb():
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


class ImprovedMultiModalBlocker:
    """Enhanced Multi-Modal Union Blocker integrating Phase 4 validated retrieval mechanisms."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_candidates = config.BLOCKING_MAX_CANDIDATES
        self.retention_policy = getattr(config, "BLOCKING_RETENTION_POLICY", "tiered").lower()
        
        # Sub-indexes
        self.exact_name_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.brand_clean_index = defaultdict(list)
        self.digit_index = defaultdict(list)
        self.compound_digit_index = defaultdict(list)
        self.low_collision_digit_index = defaultdict(list)
        self.rare_locality_token_index = defaultdict(list)
        self.address_token_index = defaultdict(list)
        
        # Enhanced TF-IDF Char (2, 3)-gram
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2)
        self.target_char_matrix = None
        self.target_ids: List[str] = []
        self.target_records: Dict[str, dict] = {}

    def build_index(self, df_targets: pd.DataFrame):
        print(f"[ImprovedBlocker] Indexing {len(df_targets)} target records across enhanced modalities...")
        
        token_freq = Counter()
        addr_token_freq = Counter()
        dig_freq = Counter()
        
        for _, row in df_targets.iterrows():
            name = row["norm_name"]
            addr = row["norm_address"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            for t in set(name.split()):
                if len(t) >= 3 and t not in COMMON_STOP_TOKENS:
                    token_freq[t] += 1
            for t in set(addr.split()):
                if len(t) >= 4 and t not in ADDRESS_STOP_TOKENS:
                    addr_token_freq[t] += 1
            for d in digits[:3]:
                if len(d) >= 2:
                    dig_freq[(d, country)] += 1

        name_cap = self.config.SMOKE_TOKEN_DOC_FREQ if self.config.LOCAL_SMOKE_TEST else min(
            self.config.MAX_TOKEN_DOC_FREQ, max(50, int(len(df_targets) * self.config.CORPUS_RELATIVE_CAP))
        )
        addr_cap = name_cap
        dig_cap = name_cap * 3
        
        valid_name_tokens = {t for t, c in token_freq.items() if c <= name_cap}
        valid_addr_tokens = {t for t, c in addr_token_freq.items() if c <= addr_cap}
        valid_digits = {k for k, c in dig_freq.items() if c <= dig_cap}
        
        names_list = []
        for _, row in df_targets.iterrows():
            eid = row["entity_id"]
            name = row["norm_name"]
            raw_name = row["business_name"]
            addr = row["norm_address"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            self.target_records[eid] = row.to_dict()
            self.target_ids.append(eid)
            names_list.append(name if name else "")
            
            # Blocker A: Exact Name
            if name:
                self.exact_name_index[name].append(eid)
            # Blocker B: Rare Name Tokens
            for t in set(name.split()):
                if t in valid_name_tokens:
                    self.token_index[t].append(eid)
            # Blocker D: Domain / Brand
            core_brand = extract_core_brand(raw_name)
            if len(core_brand) >= 4:
                self.brand_clean_index[core_brand].append(eid)
            # Blocker E: Standard Digits (len >= 3)
            for d in digits[:3]:
                if len(d) >= 3 and (d, country) in valid_digits:
                    self.digit_index[(d, country)].append(eid)
            # Enhancement D: Low-Collision Digits (len >= 2)
            for d in digits:
                if len(d) >= 2:
                    self.low_collision_digit_index[(d, country)].append(eid)
            # Compound 2-Digit Keys
            addr_distinct_tokens = [t for t in set(addr.split()) if len(t) >= 4 and t in valid_addr_tokens]
            for d in digits:
                if len(d) == 2:
                    for tok in addr_distinct_tokens[:3]:
                        self.compound_digit_index[(d, tok, country)].append(eid)
            # Blocker F: Informative Address Tokens
            for t in addr_distinct_tokens:
                self.address_token_index[t].append(eid)
            # Enhancement E: Rare Locality Tokens (len >= 5)
            for t in set(addr.split()):
                if len(t) >= 5 and t not in ADDRESS_STOP_TOKENS:
                    self.rare_locality_token_index[(t, country)].append(eid)

        # Enhancement A: Fit Char (2, 3)-gram TF-IDF Matrix
        print("[ImprovedBlocker] Vectorizing subword character (2, 3)-grams for nearest neighbors...")
        self.target_char_matrix = self.vectorizer.fit_transform(names_list)
        print(f"[ImprovedBlocker] Indexing complete. Vocabulary size: {self.target_char_matrix.shape[1]:,}")

    def generate_candidates(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
        candidates_map = {}
        s1_names = df_s1["norm_name"].fillna("").tolist()
        s1_char_matrix = self.vectorizer.transform(s1_names)
        
        chunk_size = 500
        n_s1 = len(df_s1)
        
        for start_idx in range(0, n_s1, chunk_size):
            end_idx = min(start_idx + chunk_size, n_s1)
            chunk_s1_matrix = s1_char_matrix[start_idx:end_idx]
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
                cand_modalities = defaultdict(set)
                
                # --- Modality A: Exact Normalized Name ---
                if name and name in self.exact_name_index:
                    for tid in self.exact_name_index[name]:
                        combined_cands[tid] += 50.0
                        cand_modalities[tid].add("exact_name")
                        
                # --- Modality B: Rare Name Tokens ---
                tokens = [t for t in set(name.split()) if len(t) >= 3 and t not in COMMON_STOP_TOKENS]
                for t in tokens:
                    if t in self.token_index:
                        postings = self.token_index[t]
                        w = 1.0 + (10.0 / (len(postings) + 1))
                        for tid in postings:
                            combined_cands[tid] += w
                            cand_modalities[tid].add("rare_token")
                            
                # --- Modality C: Char (2, 3)-Gram Cosine Nearest Neighbors ---
                row_vec = sim_matrix[offset]
                if row_vec.nnz > 0:
                    above_min = np.where(row_vec.data >= 0.40)[0]  # relaxed from 0.45 to 0.40
                    for idx in above_min:
                        col_idx = row_vec.indices[idx]
                        score = float(row_vec.data[idx])
                        tid = self.target_ids[col_idx]
                        combined_cands[tid] += (score * 15.0)
                        cand_modalities[tid].add("char_ngram")
                        
                # --- Modality D: Domain / Clean Brand ---
                core_brand = extract_core_brand(raw_name)
                if len(core_brand) >= 4 and core_brand in self.brand_clean_index:
                    for tid in self.brand_clean_index[core_brand]:
                        combined_cands[tid] += 25.0
                        cand_modalities[tid].add("domain_brand")
                        
                # --- Modality E: Address Digits (len >= 3) ---
                for d in digits[:3]:
                    if len(d) >= 3 and (d, country) in self.digit_index:
                        postings = self.digit_index[(d, country)]
                        w = 2.0 + (5.0 / (len(postings) + 1))
                        for tid in postings:
                            combined_cands[tid] += w
                            cand_modalities[tid].add("address_digits")
                            
                # --- Enhancement D: Low-Collision Digits (len >= 2, postings <= 30) ---
                for d in digits:
                    if len(d) >= 2:
                        key = (d, country)
                        if key in self.low_collision_digit_index:
                            postings = self.low_collision_digit_index[key]
                            if len(postings) <= 30:
                                w = 3.0 + (5.0 / (len(postings) + 1))
                                for tid in postings:
                                    combined_cands[tid] += w
                                    cand_modalities[tid].add("low_collision_digits")
                                    
                # --- Compound 2-Digit Keys ---
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
                                        cand_modalities[tid].add("compound_digits")
                                        
                # --- Modality F: Informative Address Tokens (overlap >= 2) ---
                addr_cand_counts = Counter()
                for tok in addr_tokens:
                    if tok in self.address_token_index:
                        for tid in self.address_token_index[tok]:
                            addr_cand_counts[tid] += 1
                for tid, count in addr_cand_counts.items():
                    if count >= 2:
                        combined_cands[tid] += (count * 2.5)
                        cand_modalities[tid].add("address_tokens")
                        
                # --- Enhancement E: Rare Locality Tokens (len >= 5, postings <= 20) ---
                locality_tokens = [t for t in set(addr.split()) if len(t) >= 5 and t not in ADDRESS_STOP_TOKENS]
                for tok in locality_tokens:
                    key = (tok, country)
                    if key in self.rare_locality_token_index:
                        postings = self.rare_locality_token_index[key]
                        if len(postings) <= 20:
                            w = 4.0 + (5.0 / (len(postings) + 1))
                            for tid in postings:
                                combined_cands[tid] += w
                                cand_modalities[tid].add("rare_locality")
                                
                # --- Tiered Retention Selection ---
                if not combined_cands:
                    candidates_map[s1_id] = []
                elif self.retention_policy == "tiered":
                    tier1 = []
                    tier2 = []
                    for cid, score in combined_cands.most_common():
                        m = cand_modalities[cid]
                        # Tier 1: exact name, domain/brand, compound digits, rare locality, multi-modality consensus, or confidence >= 15
                        if (
                            "exact_name" in m
                            or "domain_brand" in m
                            or "compound_digits" in m
                            or "rare_locality" in m
                            or len(m) >= 2
                            or score >= 15.0
                        ):
                            tier1.append(cid)
                        else:
                            tier2.append(cid)
                            
                    selected = tier1[:self.max_candidates]
                    remaining = self.max_candidates - len(selected)
                    if remaining > 0:
                        selected.extend(tier2[:remaining])
                    candidates_map[s1_id] = selected
                else:
                    candidates_map[s1_id] = [cid for cid, _ in combined_cands.most_common(self.max_candidates)]
                    
        return candidates_map


def validate_improved_retrieval():
    print("=" * 90)
    print("VALIDATING INTEGRATED IMPROVED RETRIEVAL (10,000 S1 UNIVERSE)")
    print("=" * 90)
    
    tracemalloc.start()
    t0 = time.time()
    
    cfg = PipelineConfig(
        DATA_ROOT=os.path.join(PROJECT_ROOT, "dataset"),
        LOCAL_SMOKE_TEST=True,
        SMOKE_SAMPLE_SIZE=10000,
        BLOCKING_MODE="multimodal",
        BLOCKING_RETENTION_POLICY="tiered",
        BLOCKING_MAX_CANDIDATES=85
    )
    
    df_s1, df_s2, df_s3, gt_train = load_train_data(cfg)
    df_s1 = normalize_dataframe(df_s1)
    df_s2 = normalize_dataframe(df_s2)
    df_s3 = normalize_dataframe(df_s3)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    n_targets = len(df_targets)
    n_s1 = len(df_s1)
    total_true_matches = sum(len(m) for m in gt_train.values())
    
    blocker = ImprovedMultiModalBlocker(cfg)
    blocker.build_index(df_targets)
    
    t_gen = time.time()
    candidates = blocker.generate_candidates(df_s1)
    gen_time = time.time() - t_gen
    
    m = evaluate_blocking(candidates, gt_train, n_targets)
    
    print("\n--- IMPROVED RETRIEVAL VALIDATION RESULTS (10,000 S1) ---")
    print(f"Blocking Recall:       {m['blocking_recall']*100:.2f}% ({m['captured_true_matches']:,}/{total_true_matches:,})")
    print(f"Captured True Matches: {m['captured_true_matches']:,} / {total_true_matches:,}")
    print(f"Lost True Matches:     {total_true_matches - m['captured_true_matches']:,} (down from 1,751 in Phase 3.75 baseline)")
    print(f"Total Candidate Pairs: {m['total_candidates']:,}")
    print(f"Mean Candidates / S1:  {m['mean_candidates_per_s1']:.1f}")
    print(f"P95 Candidates / S1:   {m['p95_candidates_per_s1']:.1f}")
    print(f"Max Candidates / S1:   {m['max_candidates_per_s1']}")
    print(f"Reduction Ratio:       {m['reduction_ratio']*100:.5f}%")
    print(f"Peak RAM:              {get_peak_mem_mb():.1f} MB")
    print(f"Candidate Gen Time:    {gen_time:.1f}s | Total Time: {time.time() - t0:.1f}s")
    
    return m, blocker, df_s1, df_targets, gt_train


if __name__ == "__main__":
    validate_improved_retrieval()

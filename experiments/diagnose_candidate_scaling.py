"""Phase 3.5 Scaling Diagnosis: Comprehensive candidate generation analysis.

Investigates recall scaling degradation across validation universes (500, 5000, 10000 S1):
1. Code inspection & candidate lifecycle audit
2. Explicit cap verification (65 ceiling)
3. Loss categorization:
   (a) Never retrieved by any blocker
   (b) Retrieved but truncated by candidate cap
   (c) Removed by other filtering rules
4. Per-blocker recall breakdown (standalone & unique marginal)
5. Recall before capping (uncapped) vs. after capping
6. Cap sweep: [50, 65, 100, 150, 250]
7. Resource & practicality evaluation (recall, volume, P95, max, RAM, runtime)
8. Dynamic per-S1 candidate capping
9. Tiered retention: preserve high-confidence candidates first
"""

import os
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe
from src.blocking import extract_core_brand, COMMON_STOP_TOKENS, ADDRESS_STOP_TOKENS

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def get_peak_mem_mb():
    current, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


class DetailedDiagnosticBlocker:
    """MultiModal blocker instrumented to record per-blocker candidate sets and ranking."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.exact_name_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.brand_clean_index = defaultdict(list)
        self.digit_index = defaultdict(list)
        self.compound_digit_index = defaultdict(list)
        self.address_token_index = defaultdict(list)
        
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
        self.target_char_matrix = None
        self.target_ids: List[str] = []
        self.target_records: Dict[str, dict] = {}
        
        # Filtering diagnostics
        self.name_doc_freq = Counter()
        self.addr_doc_freq = Counter()
        self.dig_doc_freq = Counter()
        self.name_cap = 0
        self.addr_cap = 0
        self.dig_cap = 0
        self.filtered_out_tokens = set()

    def build_index(self, df_targets: pd.DataFrame):
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

        self.name_cap = self.config.SMOKE_TOKEN_DOC_FREQ if self.config.LOCAL_SMOKE_TEST else min(
            self.config.MAX_TOKEN_DOC_FREQ, max(50, int(len(df_targets) * self.config.CORPUS_RELATIVE_CAP))
        )
        self.addr_cap = self.name_cap
        self.dig_cap = self.name_cap * 3
        
        self.name_doc_freq = token_freq
        self.addr_doc_freq = addr_token_freq
        self.dig_doc_freq = dig_freq
        
        valid_name_tokens = {t for t, c in token_freq.items() if c <= self.name_cap}
        valid_addr_tokens = {t for t, c in addr_token_freq.items() if c <= self.addr_cap}
        valid_digits = {k for k, c in dig_freq.items() if c <= self.dig_cap}
        
        self.filtered_out_tokens = {t for t, c in token_freq.items() if c > self.name_cap}
        
        names_list = []
        for _, row in df_targets.iterrows():
            eid = row["entity_id"]
            name = row["norm_name"]
            raw_name = row["business_name"]
            addr = row["norm_address"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
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
            
            # Blocker A
            if name:
                self.exact_name_index[name].append(eid)
            # Blocker B
            for t in set(name.split()):
                if t in valid_name_tokens:
                    self.token_index[t].append(eid)
            # Blocker D
            core_brand = extract_core_brand(raw_name)
            if len(core_brand) >= 4:
                self.brand_clean_index[core_brand].append(eid)
            # Blocker E
            for d in digits[:3]:
                if (d, country) in valid_digits:
                    self.digit_index[(d, country)].append(eid)
            # Compound digits
            addr_distinct_tokens = [t for t in set(addr.split()) if len(t) >= 4 and t in valid_addr_tokens]
            for d in digits:
                if len(d) == 2:
                    for tok in addr_distinct_tokens[:3]:
                        self.compound_digit_index[(d, tok, country)].append(eid)
            # Blocker F
            for t in addr_distinct_tokens:
                self.address_token_index[t].append(eid)

        self.target_char_matrix = self.vectorizer.fit_transform(names_list)

    def generate_detailed_candidates(self, df_s1: pd.DataFrame):
        """Returns per-S1 dictionary of:
        - raw combined candidates Counter {cid: score}
        - per-blocker retrieved candidate sets
        """
        s1_names = df_s1["norm_name"].fillna("").tolist()
        s1_char_matrix = self.vectorizer.transform(s1_names)
        
        results = {}
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
                
                blocker_cands = {
                    "exact_name": set(),
                    "rare_token": set(),
                    "char_3gram": set(),
                    "domain_brand": set(),
                    "address_digits": set(),
                    "compound_digits": set(),
                    "address_tokens": set()
                }
                combined_cands = Counter()
                candidate_modalities = defaultdict(set)
                
                # Blocker A: Exact Name
                if name and name in self.exact_name_index:
                    for tid in self.exact_name_index[name]:
                        blocker_cands["exact_name"].add(tid)
                        combined_cands[tid] += 50.0
                        candidate_modalities[tid].add("exact_name")
                        
                # Blocker B: Rare Name Tokens
                tokens = [t for t in set(name.split()) if len(t) >= self.config.MIN_TOKEN_LEN and t not in COMMON_STOP_TOKENS]
                for t in tokens:
                    if t in self.token_index:
                        postings = self.token_index[t]
                        w = 1.0 + (10.0 / (len(postings) + 1))
                        for tid in postings:
                            blocker_cands["rare_token"].add(tid)
                            combined_cands[tid] += w
                            candidate_modalities[tid].add("rare_token")
                            
                # Blocker C: Char 3-gram TF-IDF Cosine
                row_vec = sim_matrix[offset]
                if row_vec.nnz > 0:
                    above_min = np.where(row_vec.data >= self.config.NAME_TFIDF_MIN_SIM)[0]
                    for idx in above_min:
                        col_idx = row_vec.indices[idx]
                        score = float(row_vec.data[idx])
                        tid = self.target_ids[col_idx]
                        blocker_cands["char_3gram"].add(tid)
                        combined_cands[tid] += (score * 15.0)
                        candidate_modalities[tid].add("char_3gram")
                        
                # Blocker D: Domain / Brand
                core_brand = extract_core_brand(raw_name)
                if len(core_brand) >= 4 and core_brand in self.brand_clean_index:
                    for tid in self.brand_clean_index[core_brand]:
                        blocker_cands["domain_brand"].add(tid)
                        combined_cands[tid] += 25.0
                        candidate_modalities[tid].add("domain_brand")
                        
                # Blocker E: Digits >= 3
                for d in digits[:3]:
                    if len(d) >= 3 and (d, country) in self.digit_index:
                        postings = self.digit_index[(d, country)]
                        w = 2.0 + (5.0 / (len(postings) + 1))
                        for tid in postings:
                            blocker_cands["address_digits"].add(tid)
                            combined_cands[tid] += w
                            candidate_modalities[tid].add("address_digits")
                            
                # Blocker E2: Compound 2-digit
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
                                        blocker_cands["compound_digits"].add(tid)
                                        combined_cands[tid] += w
                                        candidate_modalities[tid].add("compound_digits")
                                        
                # Blocker F: Informative Address Tokens (>= 2 overlaps)
                addr_cand_counts = Counter()
                for tok in addr_tokens:
                    if tok in self.address_token_index:
                        for tid in self.address_token_index[tok]:
                            addr_cand_counts[tid] += 1
                for tid, count in addr_cand_counts.items():
                    if count >= 2:
                        blocker_cands["address_tokens"].add(tid)
                        combined_cands[tid] += (count * 2.5)
                        candidate_modalities[tid].add("address_tokens")
                        
                results[s1_id] = {
                    "combined_cands": combined_cands,
                    "blocker_cands": blocker_cands,
                    "candidate_modalities": candidate_modalities,
                    "s1_name": name,
                    "s1_raw_name": raw_name,
                    "s1_addr": addr,
                    "s1_digits": digits
                }
        return results


def run_scaling_diagnosis(n_s1_list: List[int]):
    caps_to_test = [50, 65, 100, 150, 250]
    
    for n_s1 in n_s1_list:
        print("\n" + "=" * 80)
        print(f"RUNNING SCALING DIAGNOSIS FOR N_S1 = {n_s1}")
        print("=" * 80)
        
        t0 = time.time()
        tracemalloc.reset_peak()
        
        cfg = PipelineConfig(
            DATA_ROOT=os.path.join(PROJECT_ROOT, "dataset"),
            LOCAL_SMOKE_TEST=True,
            SMOKE_SAMPLE_SIZE=n_s1,
            BLOCKING_MODE="multimodal"
        )
        
        # 1. Load data
        df_s1, df_s2, df_s3, gt_train = load_train_data(cfg)
        df_s1 = normalize_dataframe(df_s1)
        df_s2 = normalize_dataframe(df_s2)
        df_s3 = normalize_dataframe(df_s3)
        df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
        n_targets = len(df_targets)
        
        print(f"Loaded and normalized {len(df_s1)} S1 and {n_targets} Targets in {time.time() - t0:.1f}s.")
        print(f"Target distribution: S2={len(df_s2)}, S3={len(df_s3)}")
        
        total_true_matches = sum(len(matches) for matches in gt_train.values())
        print(f"Total True Matches in Ground Truth: {total_true_matches}")
        
        # 2. Build index
        t1 = time.time()
        diag_blocker = DetailedDiagnosticBlocker(cfg)
        diag_blocker.build_index(df_targets)
        print(f"Detailed index built in {time.time() - t1:.1f}s. Vocab caps: name={diag_blocker.name_cap}, addr={diag_blocker.addr_cap}, dig={diag_blocker.dig_cap}")
        
        # 3. Generate detailed candidates
        t2 = time.time()
        detailed_results = diag_blocker.generate_detailed_candidates(df_s1)
        gen_time = time.time() - t2
        print(f"Generated candidates in {gen_time:.1f}s.")
        
        # -------------------------------------------------------------
        # TASK 4: Breakdown of blocking recall by blocker (Standalone & Unique)
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(f"TASK 4: PER-BLOCKER RECALL BREAKDOWN (N_S1 = {n_s1})")
        print("-" * 70)
        
        blocker_keys = [
            "exact_name", "rare_token", "char_3gram", "domain_brand",
            "address_digits", "compound_digits", "address_tokens"
        ]
        
        # Calculate standalone captures and unique captures
        standalone_captures = {k: 0 for k in blocker_keys}
        unique_captures = {k: 0 for k in blocker_keys}
        
        for s1_id, true_matches in gt_train.items():
            if not true_matches:
                continue
            res = detailed_results[s1_id]
            b_cands = res["blocker_cands"]
            
            for tm in true_matches:
                retrieved_by = [k for k in blocker_keys if tm in b_cands[k]]
                for k in retrieved_by:
                    standalone_captures[k] += 1
                if len(retrieved_by) == 1:
                    unique_captures[retrieved_by[0]] += 1
                    
        print(f"{'Blocker Modality':<25} | {'Captured':<10} | {'Standalone Recall':<18} | {'Unique Matches':<15} | {'Unique %':<10}")
        print("-" * 85)
        for k in blocker_keys:
            cap_cnt = standalone_captures[k]
            st_rec = cap_cnt / total_true_matches * 100
            uniq_cnt = unique_captures[k]
            uniq_pct = uniq_cnt / total_true_matches * 100
            print(f"{k:<25} | {cap_cnt:<10} | {st_rec:>6.2f}%            | {uniq_cnt:<15} | {uniq_pct:>6.2f}%")
            
        # -------------------------------------------------------------
        # TASK 3 & 5: Raw Uncapped Recall vs. Capped Recall & Loss Audit
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(f"TASKS 3 & 5: RAW (UNCAPPED) VS CAPPED RECALL & LOSS AUDIT (N_S1 = {n_s1})")
        print("-" * 70)
        
        # Determine uncapped recall
        uncapped_captured = 0
        never_retrieved = 0
        never_retrieved_examples = []
        
        # Collect candidate lengths and ranks of true matches
        uncapped_cand_counts = []
        true_match_ranks = []  # ranks of true matches when retrieved (1-indexed)
        
        for s1_id, true_matches in gt_train.items():
            res = detailed_results[s1_id]
            comb = res["combined_cands"]
            uncapped_cand_counts.append(len(comb))
            
            # Sorted candidate list by score descending
            ranked_cands = [cid for cid, _ in comb.most_common()]
            cand_rank_map = {cid: r + 1 for r, cid in enumerate(ranked_cands)}
            
            for tm in true_matches:
                if tm in cand_rank_map:
                    uncapped_captured += 1
                    true_match_ranks.append(cand_rank_map[tm])
                else:
                    never_retrieved += 1
                    if len(never_retrieved_examples) < 10:
                        never_retrieved_examples.append({
                            "s1_id": s1_id,
                            "tm_id": tm,
                            "s1_name": res["s1_name"],
                            "s1_addr": res["s1_addr"],
                            "target_info": diag_blocker.target_records.get(tm, {})
                        })
                        
        uncapped_recall = uncapped_captured / total_true_matches * 100
        print(f"\n[Raw Uncapped Retrieval]")
        print(f"Total True Matches:           {total_true_matches}")
        print(f"Captured by Union (Uncapped): {uncapped_captured} / {total_true_matches} ({uncapped_recall:.2f}%)")
        print(f"Never Retrieved by Any Blocker (Loss Category A): {never_retrieved} ({never_retrieved / total_true_matches * 100:.2f}%)")
        
        arr_uncapped = np.array(uncapped_cand_counts)
        print(f"Uncapped Candidates per S1: Mean={np.mean(arr_uncapped):.1f}, Median={np.median(arr_uncapped):.1f}, P90={np.percentile(arr_uncapped, 90):.1f}, P95={np.percentile(arr_uncapped, 95):.1f}, P99={np.percentile(arr_uncapped, 99):.1f}, Max={np.max(arr_uncapped)}")

        # -------------------------------------------------------------
        # TASK 6 & 7: Test Candidate Caps (50, 65, 100, 150, 250)
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(f"TASKS 6 & 7: CANDIDATE CAP SWEEP RESULTS (N_S1 = {n_s1})")
        print("-" * 70)
        print(f"{'Cap':<6} | {'Recall':<10} | {'Captured':<11} | {'Lost by Cap':<12} | {'Total Cands':<12} | {'Mean/S1':<8} | {'P95':<6} | {'Max':<5} | {'Red. Ratio':<12}")
        print("-" * 95)
        
        cap_results = {}
        for cap in caps_to_test:
            captured_at_cap = 0
            lost_at_cap = 0
            cand_counts_at_cap = []
            
            for s1_id, true_matches in gt_train.items():
                res = detailed_results[s1_id]
                cands_at_cap = [cid for cid, _ in res["combined_cands"].most_common(cap)]
                cands_set = set(cands_at_cap)
                cand_counts_at_cap.append(len(cands_set))
                
                for tm in true_matches:
                    if tm in cands_set:
                        captured_at_cap += 1
                    elif tm in res["combined_cands"]:
                        lost_at_cap += 1
                        
            rec = captured_at_cap / total_true_matches * 100
            tot_cands = sum(cand_counts_at_cap)
            search_space = len(gt_train) * n_targets
            red_ratio = (1.0 - tot_cands / search_space) * 100
            arr_c = np.array(cand_counts_at_cap)
            
            cap_results[cap] = {
                "recall": rec,
                "captured": captured_at_cap,
                "lost_by_cap": lost_at_cap,
                "total_cands": tot_cands,
                "mean": np.mean(arr_c),
                "p95": np.percentile(arr_c, 95),
                "max": np.max(arr_c),
                "red_ratio": red_ratio
            }
            print(f"{cap:<6} | {rec:>6.2f}%    | {captured_at_cap:>5}/{total_true_matches} | {lost_at_cap:<12} | {tot_cands:<12} | {np.mean(arr_c):>6.1f} | {np.percentile(arr_c, 95):>5.1f} | {np.max(arr_c):<5} | {red_ratio:>9.4f}%")

        # -------------------------------------------------------------
        # Rank distribution of matches lost at cap=65
        # -------------------------------------------------------------
        ranks_lost_at_65 = [r for r in true_match_ranks if r > 65]
        print(f"\n[Rank Distribution of True Matches Lost at Cap=65]")
        print(f"Total matches lost by cap 65: {len(ranks_lost_at_65)}")
        if ranks_lost_at_65:
            arr_ranks = np.array(ranks_lost_at_65)
            print(f"Ranks: Min={np.min(arr_ranks)}, Median={np.median(arr_ranks):.1f}, P75={np.percentile(arr_ranks, 75):.1f}, P90={np.percentile(arr_ranks, 90):.1f}, Max={np.max(arr_ranks)}")
            print(f"Lost matches with rank 66-100:  {np.sum((arr_ranks >= 66) & (arr_ranks <= 100))} ({np.sum((arr_ranks >= 66) & (arr_ranks <= 100)) / len(arr_ranks) * 100:.1f}%)")
            print(f"Lost matches with rank 101-150: {np.sum((arr_ranks >= 101) & (arr_ranks <= 150))} ({np.sum((arr_ranks >= 101) & (arr_ranks <= 150)) / len(arr_ranks) * 100:.1f}%)")
            print(f"Lost matches with rank 151-250: {np.sum((arr_ranks >= 151) & (arr_ranks <= 250))} ({np.sum((arr_ranks >= 151) & (arr_ranks <= 250)) / len(arr_ranks) * 100:.1f}%)")
            print(f"Lost matches with rank > 250:   {np.sum(arr_ranks > 250)} ({np.sum(arr_ranks > 250) / len(arr_ranks) * 100:.1f}%)")

        # -------------------------------------------------------------
        # TASK 10: Test Tiered Retention Policy
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(f"TASK 10: TIERED CANDIDATE RETENTION POLICY EVALUATION (N_S1 = {n_s1})")
        print("-" * 70)
        print("Strategy: Prioritize High-Confidence matches first (Exact Name, Domain, Compound Digits, or multi-modality), then backfill with lower-scoring single-modality candidates up to cap.")
        
        for cap in [65, 100, 150]:
            captured_tiered = 0
            lost_tiered = 0
            cand_counts_tiered = []
            
            for s1_id, true_matches in gt_train.items():
                res = detailed_results[s1_id]
                comb = res["combined_cands"]
                mods = res["candidate_modalities"]
                
                # Split into Tier 1 (High Confidence) and Tier 2 (Remaining)
                tier1 = []
                tier2 = []
                
                for cid, score in comb.most_common():
                    m = mods[cid]
                    # High confidence: exact name, domain/brand, compound digit, or matches 2+ modalities, or score >= 15
                    if "exact_name" in m or "domain_brand" in m or "compound_digits" in m or len(m) >= 2 or score >= 15.0:
                        tier1.append(cid)
                    else:
                        tier2.append(cid)
                        
                # Take all Tier 1 up to cap, then backfill with Tier 2
                selected = tier1[:cap]
                remaining_slots = cap - len(selected)
                if remaining_slots > 0:
                    selected.extend(tier2[:remaining_slots])
                    
                selected_set = set(selected)
                cand_counts_tiered.append(len(selected_set))
                
                for tm in true_matches:
                    if tm in selected_set:
                        captured_tiered += 1
                    elif tm in comb:
                        lost_tiered += 1
                        
            rec_t = captured_tiered / total_true_matches * 100
            arr_t = np.array(cand_counts_tiered)
            print(f"Tiered Cap {cap:<3}: Recall = {rec_t:>6.2f}% ({captured_tiered}/{total_true_matches}) | Lost by Cap = {lost_tiered:<4} | Mean Cands = {np.mean(arr_t):.1f} | P95 = {np.percentile(arr_t, 95):.1f}")

        # -------------------------------------------------------------
        # TASK 9: Test Dynamic Per-S1 Candidate Capping
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(f"TASK 9: DYNAMIC PER-S1 CANDIDATE CAPPING (N_S1 = {n_s1})")
        print("-" * 70)
        print("Strategy: Dynamic cap based on S1 signal diversity: If S1 has strong Tier 1 matches or diverse signals, allocate up to 100/120; if low/single token, cap at 40.")
        
        # Test dynamic policy:
        # Base cap = 50. If S1 has multiple high-signal modalities or large pool of strong candidates, allow cap to expand up to 120.
        captured_dyn = 0
        lost_dyn = 0
        cand_counts_dyn = []
        
        for s1_id, true_matches in gt_train.items():
            res = detailed_results[s1_id]
            comb = res["combined_cands"]
            b_cands = res["blocker_cands"]
            
            # Compute dynamic cap for this S1
            active_modalities = sum(1 for k, v in b_cands.items() if len(v) > 0)
            if active_modalities >= 4:
                dyn_cap = 120
            elif active_modalities >= 3:
                dyn_cap = 90
            elif active_modalities >= 2:
                dyn_cap = 65
            else:
                dyn_cap = 40
                
            selected = [cid for cid, _ in comb.most_common(dyn_cap)]
            selected_set = set(selected)
            cand_counts_dyn.append(len(selected_set))
            
            for tm in true_matches:
                if tm in selected_set:
                    captured_dyn += 1
                elif tm in comb:
                    lost_dyn += 1
                    
        rec_d = captured_dyn / total_true_matches * 100
        arr_d = np.array(cand_counts_dyn)
        tot_d = sum(cand_counts_dyn)
        search_space = len(gt_train) * n_targets
        red_ratio_d = (1.0 - tot_d / search_space) * 100
        print(f"Dynamic Cap: Recall = {rec_d:>6.2f}% ({captured_dyn}/{total_true_matches}) | Lost by Cap = {lost_dyn} | Total Cands = {tot_d} | Mean = {np.mean(arr_d):.1f} | P95 = {np.percentile(arr_d, 95):.1f} | Red Ratio = {red_ratio_d:.4f}%")
        
        print(f"\nCompleted N_S1 = {n_s1} in {time.time() - t0:.1f}s. Peak RAM: {get_peak_mem_mb():.1f} MB.")


if __name__ == "__main__":
    tracemalloc.start()
    # Run on 5,000 S1 first, then 10,000 S1
    run_scaling_diagnosis([5000, 10000])

"""Phase 4.2 & 4.3: Systematic Retrieval Refinement Experiments & Unique Marginal Recall Analysis.

Evaluates individual candidate retrieval improvements on 10,000 S1 universe:
- Experiment A: Character n-gram variants (3-gram, 3+4 gram, 2+3 gram, 2+3+4 gram)
- Experiment B: Character similarity threshold sweep (0.35, 0.40, 0.45, 0.50)
- Experiment C: Alternate normalized name forms (token-sorted, punctuation-stripped, legal-cleaned)
- Experiment D: Cross-script / transliteration-aware retrieval (Indic digit extraction, script-agnostic address tokens)
- Experiment E: Soft address retrieval (relaxed address token overlap >= 1 with digit agreement, address 3-grams)
- Experiment F: Domain/handle normalization (expanded TLDs: .fr, .io, .co, .org, handle prefix @)

Measures for each:
- Standalone recall
- Unique matches recovered (unique marginal recall)
- Candidate volume / growth
- Runtime and RAM
"""

import os
import sys
import time
import tracemalloc
import re
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
from src.blocking import MultiModalCandidateBlocker, extract_core_brand, COMMON_STOP_TOKENS, ADDRESS_STOP_TOKENS

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def get_peak_mem_mb():
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def eval_sparse_similarity(
    X_s1, X_target, df_s1, df_targets, gt_train,
    baseline_captured_matches, total_true_matches, min_sim, chunk_size=500
):
    target_ids = df_targets["entity_id"].values
    s1_ids = df_s1["entity_id"].values
    n_s1 = len(s1_ids)
    
    captured = 0
    unique_recovered = 0
    cand_counts = []
    
    for start_idx in range(0, n_s1, chunk_size):
        end_idx = min(start_idx + chunk_size, n_s1)
        chunk_sim = (X_s1[start_idx:end_idx] * X_target.T).tocsr()
        
        for offset in range(end_idx - start_idx):
            s1_id = s1_ids[start_idx + offset]
            gt_m = gt_train.get(s1_id, set())
            
            row_vec = chunk_sim[offset]
            cands = set()
            if row_vec.nnz > 0:
                above_min = np.where(row_vec.data >= min_sim)[0]
                for idx in above_min:
                    cands.add(target_ids[row_vec.indices[idx]])
            cand_counts.append(len(cands))
            
            for tm in gt_m:
                if tm in cands:
                    captured += 1
                    if (s1_id, tm) not in baseline_captured_matches:
                        unique_recovered += 1
                        
    rec = captured / total_true_matches * 100
    mean_c = float(np.mean(cand_counts))
    return rec, captured, unique_recovered, mean_c


def run_retrieval_experiments():
    print("=" * 90)
    print("PHASE 4.2 & 4.3: RETRIEVAL REFINEMENT & UNIQUE MARGINAL RECALL ANALYSIS")
    print("=" * 90)
    
    tracemalloc.start()
    t_start = time.time()
    
    cfg = PipelineConfig(
        DATA_ROOT=os.path.join(PROJECT_ROOT, "dataset"),
        LOCAL_SMOKE_TEST=True,
        SMOKE_SAMPLE_SIZE=10000,
        BLOCKING_MODE="multimodal",
        BLOCKING_RETENTION_POLICY="tiered",
        BLOCKING_MAX_CANDIDATES=85
    )
    
    # 1. Load Data
    t0 = time.time()
    df_s1, df_s2, df_s3, gt_train = load_train_data(cfg)
    df_s1 = normalize_dataframe(df_s1)
    df_s2 = normalize_dataframe(df_s2)
    df_s3 = normalize_dataframe(df_s3)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    n_targets = len(df_targets)
    n_s1 = len(df_s1)
    
    total_true_matches = sum(len(m) for m in gt_train.values())
    print(f"Loaded: S1={n_s1}, Targets={n_targets}, Ground Truth Matches={total_true_matches:,} in {time.time() - t0:.1f}s")
    
    # 2. Build Standard Baseline MultiModal Blocker
    t_base = time.time()
    base_blocker = MultiModalCandidateBlocker(cfg)
    base_blocker.build_index(df_targets)
    base_candidates = base_blocker.generate_candidates(df_s1)
    
    # Find baseline captured and uncaptured ground truth matches
    baseline_captured_matches = set()
    for s1_id, true_matches in gt_train.items():
        cands = set(base_candidates.get(s1_id, []))
        for tm in true_matches:
            if tm in cands:
                baseline_captured_matches.add((s1_id, tm))
                
    base_recall = len(baseline_captured_matches) / total_true_matches * 100
    base_total_cands = sum(len(c) for c in base_candidates.values())
    base_mean_cands = base_total_cands / n_s1
    print(f"\n[Baseline Blocker (Tiered 85)] Recall: {base_recall:.2f}% ({len(baseline_captured_matches):,}/{total_true_matches:,}) | "
          f"Mean Cands/S1: {base_mean_cands:.1f} | Total Pairs: {base_total_cands:,} in {time.time() - t_base:.1f}s")
          
    results = []

    # -----------------------------------------------------------------
    # EXPERIMENT A: Character N-Gram Variants (3-gram, 3+4 gram, 2+3 gram, 2+3+4 gram)
    # -----------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EXPERIMENT A: CHARACTER N-GRAM VARIANTS")
    print("-" * 75)
    
    target_names = df_targets["norm_name"].fillna("").tolist()
    s1_names = df_s1["norm_name"].fillna("").tolist()
    ngram_ranges = [
        ("char_3gram (Baseline)", (3, 3)),
        ("char_3+4gram", (3, 4)),
        ("char_2+3gram", (2, 3)),
        ("char_2+3+4gram", (2, 4))
    ]
    
    for label, n_range in ngram_ranges:
        t_ng = time.time()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=n_range, min_df=2)
        X_target = vec.fit_transform(target_names)
        X_s1 = vec.transform(s1_names)
        
        rec, captured, unique_recovered, mean_c = eval_sparse_similarity(
            X_s1, X_target, df_s1, df_targets, gt_train,
            baseline_captured_matches, total_true_matches, min_sim=0.45, chunk_size=500
        )
        
        elapsed = time.time() - t_ng
        print(f"[{label:<22}] Recall: {rec:>6.2f}% ({captured:>5}/{total_true_matches}) | Unique Recovered: {unique_recovered:>3} | Mean Cands: {mean_c:>5.1f} | Vocab: {X_target.shape[1]:>6} | Time: {elapsed:>4.1f}s")
        results.append({
            "Experiment": "Exp A: N-Gram Variants",
            "Variant": label,
            "Recall": rec,
            "Unique_Recovered": unique_recovered,
            "Mean_Cands": mean_c,
            "Runtime_s": elapsed
        })

    # -----------------------------------------------------------------
    # EXPERIMENT B: Character Similarity Threshold Sweep (0.35, 0.40, 0.45, 0.50)
    # -----------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EXPERIMENT B: CHARACTER SIMILARITY THRESHOLD SWEEP")
    print("-" * 75)
    
    vec3 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
    X_target3 = vec3.fit_transform(target_names)
    X_s1_3 = vec3.transform(s1_names)
    
    thresholds = [0.35, 0.40, 0.45, 0.50]
    for sim_thresh in thresholds:
        t_th = time.time()
        rec, captured, unique_recovered, mean_c = eval_sparse_similarity(
            X_s1_3, X_target3, df_s1, df_targets, gt_train,
            baseline_captured_matches, total_true_matches, min_sim=sim_thresh, chunk_size=500
        )
        elapsed = time.time() - t_th
        print(f"[Sim Thresh {sim_thresh:.2f}{' (Baseline)' if sim_thresh==0.45 else ''}] Recall: {rec:>6.2f}% ({captured:>5}/{total_true_matches}) | Unique Recovered: {unique_recovered:>3} | Mean Cands: {mean_c:>5.1f} | Time: {elapsed:>4.1f}s")
        results.append({
            "Experiment": "Exp B: Similarity Threshold",
            "Variant": f"NAME_TFIDF_MIN_SIM={sim_thresh:.2f}",
            "Recall": rec,
            "Unique_Recovered": unique_recovered,
            "Mean_Cands": mean_c,
            "Runtime_s": elapsed
        })

    # -----------------------------------------------------------------
    # EXPERIMENT C: Alternate Normalized Name Forms
    # -----------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EXPERIMENT C: ALTERNATE NORMALIZED NAME FORMS")
    print("-" * 75)
    
    # 1. Token-Sorted Name index
    token_sorted_index = defaultdict(list)
    for idx, name in enumerate(target_names):
        toks = tuple(sorted(name.split()))
        if toks:
            token_sorted_index[toks].append(df_targets.iloc[idx]["entity_id"])
            
    captured_ts = 0
    unique_ts = 0
    ts_cands_count = []
    t_c = time.time()
    
    for s1_id, name in zip(df_s1["entity_id"].values, s1_names):
        gt_m = gt_train.get(s1_id, set())
        toks = tuple(sorted(name.split()))
        cands = set(token_sorted_index.get(toks, []))
        ts_cands_count.append(len(cands))
        
        for tm in gt_m:
            if tm in cands:
                captured_ts += 1
                if (s1_id, tm) not in baseline_captured_matches:
                    unique_ts += 1
                    
    rec_ts = captured_ts / total_true_matches * 100
    print(f"[Token-Sorted Exact Name] Recall: {rec_ts:>6.2f}% ({captured_ts:>5}/{total_true_matches}) | Unique Recovered: {unique_ts:>3} | Mean Cands: {np.mean(ts_cands_count):.2f} | Time: {time.time() - t_c:.1f}s")
    results.append({
        "Experiment": "Exp C: Name Normalization",
        "Variant": "Token-Sorted Exact Name",
        "Recall": rec_ts,
        "Unique_Recovered": unique_ts,
        "Mean_Cands": float(np.mean(ts_cands_count)),
        "Runtime_s": time.time() - t_c
    })

    # -----------------------------------------------------------------
    # EXPERIMENT D: Cross-Script & Transliteration Soft Digit/Token Matching
    # -----------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EXPERIMENT D: CROSS-SCRIPT / TRANSLITERATION-AWARE RETRIEVAL")
    print("-" * 75)
    # The forensic analysis showed 58.5% of misses were cross-script where S1 has digits or PIN
    # Test: Single 2-digit PIN prefix or Unit number paired with country (max collisions <= 50)
    t_d = time.time()
    indic_digit_index = defaultdict(list)
    for idx, row in df_targets.iterrows():
        eid = row["entity_id"]
        country = row["norm_country"]
        digits = row.get("address_digits", [])
        for d in digits:
            if len(d) >= 2:
                indic_digit_index[(d, country)].append(eid)
                
    captured_d = 0
    unique_d = 0
    d_counts = []
    
    for idx, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        gt_m = gt_train.get(s1_id, set())
        country = row["norm_country"]
        digits = row.get("address_digits", [])
        cands = set()
        
        for d in digits:
            if len(d) >= 2:
                postings = indic_digit_index.get((d, country), [])
                if len(postings) <= 30:  # low collision only
                    cands.update(postings)
        d_counts.append(len(cands))
        
        for tm in gt_m:
            if tm in cands:
                captured_d += 1
                if (s1_id, tm) not in baseline_captured_matches:
                    unique_d += 1
                    
    rec_d = captured_d / total_true_matches * 100
    print(f"[Low-Collision Digits (len>=2)] Recall: {rec_d:>6.2f}% ({captured_d:>5}/{total_true_matches}) | Unique Recovered: {unique_d:>3} | Mean Cands: {np.mean(d_counts):.2f} | Time: {time.time() - t_d:.1f}s")
    results.append({
        "Experiment": "Exp D: Cross-Script Digit Retrieval",
        "Variant": "Low-Collision Digits (len>=2, postings<=30)",
        "Recall": rec_d,
        "Unique_Recovered": unique_d,
        "Mean_Cands": float(np.mean(d_counts)),
        "Runtime_s": time.time() - t_d
    })

    # -----------------------------------------------------------------
    # EXPERIMENT E: Address Soft Retrieval (Relaxed Token Overlap with Digit Guard)
    # -----------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EXPERIMENT E: ADDRESS SOFT RETRIEVAL")
    print("-" * 75)
    # Address tokens >= 1 IF and only if common digits >= 1 and address token length >= 5
    t_e = time.time()
    rare_addr_token_index = defaultdict(list)
    for idx, row in df_targets.iterrows():
        eid = row["entity_id"]
        addr = row["norm_address"]
        country = row["norm_country"]
        tokens = [t for t in set(addr.split()) if len(t) >= 5 and t not in ADDRESS_STOP_TOKENS]
        for t in tokens:
            rare_addr_token_index[(t, country)].append(eid)
            
    captured_e = 0
    unique_e = 0
    e_counts = []
    
    for idx, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        gt_m = gt_train.get(s1_id, set())
        addr = row["norm_address"]
        country = row["norm_country"]
        s1_digs = set(row.get("address_digits", []))
        
        cands = set()
        tokens = [t for t in set(addr.split()) if len(t) >= 5 and t not in ADDRESS_STOP_TOKENS]
        for t in tokens:
            postings = rare_addr_token_index.get((t, country), [])
            if len(postings) <= 20:  # highly informative locality tokens
                cands.update(postings)
        e_counts.append(len(cands))
        
        for tm in gt_m:
            if tm in cands:
                captured_e += 1
                if (s1_id, tm) not in baseline_captured_matches:
                    unique_e += 1
                    
    rec_e = captured_e / total_true_matches * 100
    print(f"[Rare Locality Tokens (len>=5, <=20)] Recall: {rec_e:>6.2f}% ({captured_e:>5}/{total_true_matches}) | Unique Recovered: {unique_e:>3} | Mean Cands: {np.mean(e_counts):.2f} | Time: {time.time() - t_e:.1f}s")
    results.append({
        "Experiment": "Exp E: Address Soft Retrieval",
        "Variant": "Rare Locality Tokens (len>=5, postings<=20)",
        "Recall": rec_e,
        "Unique_Recovered": unique_e,
        "Mean_Cands": float(np.mean(e_counts)),
        "Runtime_s": time.time() - t_e
    })

    # -----------------------------------------------------------------
    # EXPERIMENT F: Domain / Handle Normalization
    # -----------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EXPERIMENT F: DOMAIN / HANDLE NORMALIZATION")
    print("-" * 75)
    # Extended regex covering .fr, .in, .org, .co, .io, .biz, .net, .com, and handle symbols
    t_f = time.time()
    clean_brand_index = defaultdict(list)
    for idx, row in df_targets.iterrows():
        raw_name = row["business_name"]
        cb = extract_core_brand(raw_name)
        if len(cb) >= 4:
            clean_brand_index[cb].append(row["entity_id"])
            
    captured_f = 0
    unique_f = 0
    f_counts = []
    
    for idx, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        gt_m = gt_train.get(s1_id, set())
        raw_name = row["business_name"]
        cb = extract_core_brand(raw_name)
        cands = set(clean_brand_index.get(cb, [])) if len(cb) >= 4 else set()
        f_counts.append(len(cands))
        
        for tm in gt_m:
            if tm in cands:
                captured_f += 1
                if (s1_id, tm) not in baseline_captured_matches:
                    unique_f += 1
                    
    rec_f = captured_f / total_true_matches * 100
    print(f"[Core Brand / Domain Match] Recall: {rec_f:>6.2f}% ({captured_f:>5}/{total_true_matches}) | Unique Recovered: {unique_f:>3} | Mean Cands: {np.mean(f_counts):.2f} | Time: {time.time() - t_f:.1f}s")
    results.append({
        "Experiment": "Exp F: Domain/Handle Match",
        "Variant": "Core Brand / Domain Match",
        "Recall": rec_f,
        "Unique_Recovered": unique_f,
        "Mean_Cands": float(np.mean(f_counts)),
        "Runtime_s": time.time() - t_f
    })

    # Save summary table
    df_exp = pd.DataFrame(results)
    out_csv = os.path.join(PROJECT_ROOT, "experiments", "phase4_retrieval_benchmarks.csv")
    df_exp.to_csv(out_csv, index=False)
    print(f"\nSaved retrieval experiments to {out_csv}")
    print(f"Total Experiment Time: {time.time() - t_start:.1f}s. Peak RAM: {get_peak_mem_mb():.1f} MB.")
    return df_exp


if __name__ == "__main__":
    run_retrieval_experiments()

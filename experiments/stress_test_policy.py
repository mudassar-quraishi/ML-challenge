"""Phase 3.75: Production-cap stress test on 5,000 S1 and 10,000 S1.

Verifies stability of Tiered 65, Tiered 85, and Tiered 100 policies:
- Recall stability
- Candidate volume scaling
- RAM & runtime
- Singleton behavior
"""

import os
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe
from src.blocking import CandidateBlocker, evaluate_blocking
from src.train import train_matching_model

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def get_peak_mem_mb():
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def run_stress_test(sample_sizes=[5000, 10000], policies=[("tiered", 85), ("tiered", 65), ("tiered", 100)]):
    print("=" * 80)
    print("PHASE 3.75: PRODUCTION-CAP STRESS TEST ACROSS 5,000 S1 & 10,000 S1")
    print("=" * 80)
    
    tracemalloc.start()
    records = []
    
    for n_s1 in sample_sizes:
        print(f"\n==========================================")
        print(f" EVALUATING N_S1 = {n_s1:,}")
        print(f"==========================================")
        
        cfg = PipelineConfig(
            DATA_ROOT=os.path.join(PROJECT_ROOT, "dataset"),
            LOCAL_SMOKE_TEST=True,
            SMOKE_SAMPLE_SIZE=n_s1,
            BLOCKING_MODE="multimodal"
        )
        
        t0 = time.time()
        df_s1, df_s2, df_s3, gt_train = load_train_data(cfg)
        df_s1 = normalize_dataframe(df_s1)
        df_s2 = normalize_dataframe(df_s2)
        df_s3 = normalize_dataframe(df_s3)
        df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
        n_targets = len(df_targets)
        load_time = time.time() - t0
        print(f"Loaded & Normalized {len(df_s1)} S1, {n_targets} Targets in {load_time:.1f}s.")
        
        # Build index once
        t_idx = time.time()
        cfg.BLOCKING_RETENTION_POLICY = "tiered"
        cfg.BLOCKING_MAX_CANDIDATES = 100
        blocker = CandidateBlocker(cfg)
        blocker.build_index(df_targets)
        print(f"Index built in {time.time() - t_idx:.1f}s.")
        
        # Prepare dictionaries for LightGBM
        s1_dict = {row["entity_id"]: row.to_dict() for _, row in df_s1.iterrows()}
        target_dict = {row["entity_id"]: row.to_dict() for _, row in df_targets.iterrows()}
        
        for mode, cap in policies:
            name = f"{mode.capitalize()} {cap}"
            print(f"\n--- Testing {name} on N_S1 = {n_s1} ---")
            t_pol = time.time()
            blocker.retention_policy = mode
            blocker.max_candidates = cap
            
            cands = blocker.generate_candidates(df_s1)
            b_m = evaluate_blocking(cands, gt_train, n_targets)
            
            # Run model
            model, thresh, val_m = train_matching_model(
                cands, gt_train, s1_dict, target_dict, cfg
            )
            
            elapsed = time.time() - t_pol
            rec = {
                "Universe_S1": n_s1,
                "Targets": n_targets,
                "Policy": name,
                "Cap": cap,
                "Blocking_Recall": b_m["blocking_recall"],
                "Captured": b_m["captured_true_matches"],
                "Total_GT": b_m["total_true_matches"],
                "Candidates": b_m["total_candidates"],
                "Mean_Cands_S1": b_m["mean_candidates_per_s1"],
                "P95_Cands_S1": b_m["p95_candidates_per_s1"],
                "Max_Cands_S1": b_m["max_candidates_per_s1"],
                "Reduction_Ratio": b_m["reduction_ratio"],
                "Val_Macro_F05": val_m["macro_f05"],
                "Val_Precision": val_m["macro_precision"],
                "Val_Recall": val_m["macro_recall"],
                "Singleton_Acc": val_m["singleton_accuracy"],
                "Threshold": thresh,
                "Runtime_s": elapsed,
                "Peak_RAM_MB": get_peak_mem_mb()
            }
            records.append(rec)
            print(f"[{name} @ {n_s1} S1] Recall: {b_m['blocking_recall']*100:.2f}% | F0.5: {val_m['macro_f05']:.4f} | "
                  f"Prec: {val_m['macro_precision']:.4f} | Rec: {val_m['macro_recall']:.4f} | SnglAcc: {val_m['singleton_accuracy']*100:.2f}% | "
                  f"MeanCands: {b_m['mean_candidates_per_s1']:.1f} | Thresh: {thresh:.2f} | Time: {elapsed:.1f}s")
                  
    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY TABLE")
    print("=" * 80)
    print(df[["Universe_S1", "Policy", "Blocking_Recall", "Mean_Cands_S1", "P95_Cands_S1", "Val_Macro_F05", "Val_Precision", "Val_Recall", "Singleton_Acc", "Threshold", "Runtime_s"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_stress_test([5000, 10000], [("tiered", 85), ("tiered", 65), ("tiered", 100)])

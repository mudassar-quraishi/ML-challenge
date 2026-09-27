"""Phase 4.6: Feature Ablation & Downstream Model Validation (10,000 S1 Universe).

Tests 4 feature configurations on the identical entity-level split:
1. Group A: Baseline Features (22 original features)
2. Group B: Baseline + Enhanced Name Features (25 features)
3. Group C: Baseline + Enhanced Address Features (26 features)
4. Group D: Full Phase 4 Feature Suite with Interactions & Consensus (34 features)

Measures for each:
- Optimal Threshold
- Validation Macro F0.5
- Validation Precision
- Validation Recall
- Singleton Accuracy
- Non-Singleton Macro F0.5
- Feature Importances (gain)
"""

import json
import os
import sys
import time
import tracemalloc
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe
from src.blocking import CandidateBlocker, evaluate_blocking
from src.features import FEATURE_NAMES, build_feature_matrix
from src.evaluate import (
    entity_level_train_val_split,
    calibrate_threshold,
    evaluate_predictions
)

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def prepare_pairs_and_labels(
    candidates_map: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    s1_ids: List[str]
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    pairs = []
    labels = []
    for s1_id in s1_ids:
        cands = candidates_map.get(s1_id, [])
        gt_matches = ground_truth.get(s1_id, set())
        for cid in cands:
            pairs.append((s1_id, cid))
            labels.append(1 if cid in gt_matches else 0)
    return pairs, np.array(labels, dtype=np.int32)


def run_feature_ablation():
    print("=" * 90)
    print("PHASE 4.6: FEATURE ABLATION & DOWNSTREAM MODEL VALIDATION (10,000 S1 UNIVERSE)")
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
    print(f"Data Loaded: S1={n_s1}, Targets={n_targets}, Ground Truth Matches={total_true_matches:,} in {time.time() - t0:.1f}s")
    
    # 2. Candidate Generation with Production Blocker
    t_blk = time.time()
    blocker = CandidateBlocker(cfg)
    blocker.build_index(df_targets)
    candidates_map = blocker.generate_candidates(df_s1)
    blk_eval = evaluate_blocking(candidates_map, gt_train, n_targets)
    print(f"Candidate Blocking Complete in {time.time() - t_blk:.1f}s:")
    print(f"  Recall: {blk_eval['blocking_recall']*100:.2f}% ({blk_eval['captured_true_matches']:,}/{total_true_matches:,})")
    print(f"  Total Candidate Pairs: {blk_eval['total_candidates']:,} (Mean: {blk_eval['mean_candidates_per_s1']:.1f}, P95: {blk_eval['p95_candidates_per_s1']:.1f})")
    print(f"  Reduction Ratio: {blk_eval['reduction_ratio']*100:.5f}%")
    
    # S1 & Target Dict Lookups
    s1_dict = {row["entity_id"]: row.to_dict() for _, row in df_s1.iterrows()}
    target_dict = blocker.target_records
    
    # 3. Entity-level Train/Val Split (80/20) with seed 42
    all_s1_ids = list(candidates_map.keys())
    train_s1, val_s1 = entity_level_train_val_split(all_s1_ids, val_ratio=0.20, seed=42)
    print(f"\nEntity-Level Split: {len(train_s1)} Train S1, {len(val_s1)} Val S1 (Zero entity leakage)")
    
    train_pairs, y_train = prepare_pairs_and_labels(candidates_map, gt_train, train_s1)
    val_pairs, y_val = prepare_pairs_and_labels(candidates_map, gt_train, val_s1)
    print(f"Train Pairs: {len(train_pairs):,} (Pos: {np.sum(y_train==1):,}, Neg: {np.sum(y_train==0):,})")
    print(f"Val Pairs:   {len(val_pairs):,} (Pos: {np.sum(y_val==1):,}, Neg: {np.sum(y_val==0):,})")
    
    # 4. Compute Full Feature Matrix Once (34 Features)
    print(f"\nComputing 34 pairwise features for {len(train_pairs):,} train pairs and {len(val_pairs):,} val pairs...")
    t_feat = time.time()
    X_train_full = build_feature_matrix(train_pairs, s1_dict, target_dict)
    X_val_full = build_feature_matrix(val_pairs, s1_dict, target_dict)
    print(f"Feature computation complete in {time.time() - t_feat:.1f}s. Matrix shape: Train={X_train_full.shape}, Val={X_val_full.shape}")
    
    # Define Feature Ablation Configurations
    baseline_features = [
        "exact_name_match", "exact_norm_name_match", "name_token_jaccard",
        "name_token_overlap_count", "name_length_diff", "name_length_ratio",
        "name_fuzz_ratio", "name_fuzz_wratio", "name_token_sort_ratio", "name_token_set_ratio",
        "address_exact_match", "address_norm_match", "address_token_jaccard",
        "address_token_overlap_count", "address_digit_jaccard", "address_digit_overlap",
        "address_is_empty_target", "address_fuzz_ratio", "country_exact_match",
        "name_ratio_x_country", "name_ratio_x_address_jaccard", "name_exact_x_country"
    ]
    name_additions = ["name_jaro_winkler", "name_prefix_common_len", "brand_core_match"]
    addr_additions = ["address_token_sort_ratio", "address_digit_overlap_count", "address_pin_match", "address_street_number_match"]
    full_features = FEATURE_NAMES
    
    ablation_groups = [
        ("Group A: Baseline Features (22)", baseline_features),
        ("Group B: Baseline + Name Additions (25)", baseline_features + name_additions),
        ("Group C: Baseline + Address Additions (26)", baseline_features + addr_additions),
        ("Group D: Full Suite with Interactions (34)", full_features),
    ]
    
    val_gt = {s1_id: gt_train.get(s1_id, set()) for s1_id in val_s1}
    ablation_results = []
    
    print("\n" + "=" * 90)
    print("RUNNING FEATURE ABLATION EXPERIMENTS")
    print("=" * 90)
    
    for group_name, feat_subset in ablation_groups:
        col_indices = [FEATURE_NAMES.index(fn) for fn in feat_subset]
        X_tr = X_train_full[:, col_indices]
        X_va = X_val_full[:, col_indices]
        
        t_tr = time.time()
        clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            importance_type="gain",
            verbose=-1
        )
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t_tr
        
        val_probs = clf.predict_proba(X_va)[:, 1]
        
        # Group val candidate scores by S1
        val_scores: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in val_s1}
        for (s1_id, cid), p in zip(val_pairs, val_probs):
            val_scores[s1_id].append((cid, float(p)))
            
        best_thresh, best_metrics, _ = calibrate_threshold(val_scores, val_gt)
        
        # Get top 3 features by gain
        importances = clf.feature_importances_
        sorted_feats = sorted(zip(feat_subset, importances), key=lambda x: x[1], reverse=True)
        top3_str = ", ".join([f"{f} ({imp:.1f})" for f, imp in sorted_feats[:3]])
        
        print(f"\n[{group_name}]")
        print(f"  Features Count:    {len(feat_subset)}")
        print(f"  Best Threshold:    {best_thresh:.2f}")
        print(f"  Macro F0.5:        {best_metrics['macro_f05']:.4f}")
        print(f"  Macro Precision:   {best_metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:      {best_metrics['macro_recall']:.4f}")
        print(f"  Singleton Acc:     {best_metrics['singleton_accuracy']*100:.2f}% ({best_metrics['singleton_total']} singletons)")
        print(f"  Non-Singleton F0.5:{best_metrics['non_singleton_macro_f05']:.4f}")
        print(f"  Top Features:      {top3_str}")
        print(f"  Fit Time:          {fit_time:.1f}s")
        
        ablation_results.append({
            "Configuration": group_name,
            "Feature_Count": len(feat_subset),
            "Optimal_Threshold": best_thresh,
            "Macro_F05": best_metrics["macro_f05"],
            "Macro_Precision": best_metrics["macro_precision"],
            "Macro_Recall": best_metrics["macro_recall"],
            "Singleton_Accuracy": best_metrics["singleton_accuracy"],
            "Non_Singleton_F05": best_metrics["non_singleton_macro_f05"],
            "Top_Feature_1": sorted_feats[0][0],
            "Top_Feature_1_Gain": sorted_feats[0][1],
            "Top_Feature_2": sorted_feats[1][0],
            "Top_Feature_2_Gain": sorted_feats[1][1],
            "Top_Feature_3": sorted_feats[2][0],
            "Top_Feature_3_Gain": sorted_feats[2][1],
            "Fit_Time_s": fit_time
        })
        
    df_results = pd.DataFrame(ablation_results)
    out_csv = os.path.join(PROJECT_ROOT, "experiments", "phase4_experiment_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nAblation results saved to: {out_csv}")
    
    return df_results


if __name__ == "__main__":
    run_feature_ablation()

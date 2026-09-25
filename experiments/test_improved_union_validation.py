"""Validate the downstream impact of the improved Phase 2 Multi-Modal Union Blocker.

Compares:
- Baseline (EXP-001): 91.68% blocking recall, 0.9514 validation macro F0.5
- Phase 2 Multi-Modal Union (EXP-002): 98.93% blocking recall -> New validation macro F0.5
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe
from src.blocking import evaluate_blocking
from src.train import train_matching_model

from experiments.evaluate_blockers import (
    BlockerA_ExactName,
    BlockerB_RareToken,
    BlockerC_CharNgram,
    BlockerD_UrlDomain,
    BlockerE_AddressDigits,
    BlockerF_AddressTokens
)

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("EXP-002: Downstream Validation with Phase 2 Multi-Modal Union Blocker")
print("=" * 70)

config = PipelineConfig(
    DATA_ROOT=os.path.join(PROJECT_ROOT, "dataset"),
    LOCAL_SMOKE_TEST=True,
    SMOKE_SAMPLE_SIZE=500
)

# 1. Load data
df_s1, df_s2, df_s3, gt_train = load_train_data(config)
df_s1 = normalize_dataframe(df_s1)
df_s2 = normalize_dataframe(df_s2)
df_s3 = normalize_dataframe(df_s3)
df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
n_targets = len(df_targets)

# 2. Build multi-modal blockers
print("[MultiModalBlocker] Building individual indexes...")
b_a = BlockerA_ExactName()
b_b = BlockerB_RareToken(max_doc_freq=50)
b_c = BlockerC_CharNgram(min_sim=0.45, top_k=30)
b_d = BlockerD_UrlDomain()
b_e = BlockerE_AddressDigits(max_doc_freq=150)
b_f = BlockerF_AddressTokens(max_doc_freq=50)

for b in [b_a, b_b, b_c, b_d, b_e, b_f]:
    b.build(df_targets)

# 3. Query and generate union candidate map
print("[MultiModalBlocker] Generating union candidate pairs...")
union_candidates = {}
for _, r in df_s1.iterrows():
    s1_id = r["entity_id"]
    row_dict = r.to_dict()
    
    combined = Counter()
    for b in [b_a, b_b, b_c, b_d, b_e, b_f]:
        cands = b.query(row_dict)
        for rank, cid in enumerate(cands):
            score = 1.0 / (rank + 1.0)
            if "Exact" in b.name:
                score += 50.0
            elif "Url" in b.name:
                score += 20.0
            elif "Char" in b.name:
                score += 5.0
            combined[cid] += score
            
    if not combined:
        union_candidates[s1_id] = []
    else:
        union_candidates[s1_id] = [cid for cid, _ in combined.most_common(65)]

# 4. Measure blocking
blocking_m = evaluate_blocking(union_candidates, gt_train, n_targets)
print("\n--- PHASE 2 BLOCKING RESULTS ---")
print(f"Blocking Recall:  {blocking_m['blocking_recall']*100:.2f}% (Captured: {blocking_m['captured_true_matches']}/{blocking_m['total_true_matches']})")
print(f"Total Candidates: {blocking_m['total_candidates']}")
print(f"Mean Cands / S1:  {blocking_m['mean_candidates_per_s1']:.1f}")
print(f"P95 Cands / S1:   {blocking_m['p95_candidates_per_s1']:.1f}")
print(f"Reduction Ratio:  {blocking_m['reduction_ratio']*100:.4f}%")

# 5. Build target dictionary
target_records = {}
for _, row in df_targets.iterrows():
    target_records[row["entity_id"]] = {
        "business_name": row["business_name"],
        "norm_name": row["norm_name"],
        "business_address": row.get("business_address", ""),
        "norm_address": row.get("norm_address", ""),
        "country": row.get("country", ""),
        "norm_country": row["norm_country"],
        "address_digits": row.get("address_digits", [])
    }

s1_records = {row["entity_id"]: row.to_dict() for _, row in df_s1.iterrows()}

# 6. Train model and calibrate
print("\n--- RUNNING MODEL TRAINING & CALIBRATION ON IMPROVED CANDIDATES ---")
model, best_thresh, val_metrics = train_matching_model(
    union_candidates,
    gt_train,
    s1_records,
    target_records,
    config
)

print("\n" + "=" * 70)
print("EXP-001 (Baseline) vs EXP-002 (Phase 2 Multi-Modal Union)")
print("=" * 70)
print(f"Blocking Recall:       91.68%  -->  {blocking_m['blocking_recall']*100:.2f}%  (+{blocking_m['blocking_recall']*100 - 91.68:.2f}%)")
print(f"Captured Matches:      1630    -->  {blocking_m['captured_true_matches']}   (+{blocking_m['captured_true_matches'] - 1630})")
print(f"Total Candidates:      10107   -->  {blocking_m['total_candidates']}")
print(f"Mean Cands / S1:       20.2    -->  {blocking_m['mean_candidates_per_s1']:.1f}")
print(f"Calibrated Threshold:  0.40    -->  {best_thresh:.2f}")
print(f"Validation Macro F0.5: 0.9514  -->  {val_metrics['macro_f05']:.4f}")
print(f"Validation Precision:  0.9730  -->  {val_metrics['macro_precision']:.4f}")
print(f"Validation Recall:     0.9127  -->  {val_metrics['macro_recall']:.4f}")
print("=" * 70)

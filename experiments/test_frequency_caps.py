"""Evaluate the effect of candidate explosion protection / frequency caps.

Tests frequency cap thresholds for:
- Name token document frequency (10, 25, 50, 100, 250, 500, No Cap)
- Measures: blocking recall, total candidates, mean candidates/S1, max candidates/S1
"""

import os
import sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe
from src.blocking import evaluate_blocking, COMMON_STOP_TOKENS

sys.stdout.reconfigure(encoding='utf-8')

config = PipelineConfig(
    DATA_ROOT=os.path.join(PROJECT_ROOT, "dataset"),
    LOCAL_SMOKE_TEST=True,
    SMOKE_SAMPLE_SIZE=500
)

df_s1, df_s2, df_s3, gt_train = load_train_data(config)
df_s1 = normalize_dataframe(df_s1)
df_s2 = normalize_dataframe(df_s2)
df_s3 = normalize_dataframe(df_s3)
df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
n_targets = len(df_targets)
total_true = sum(len(m) for m in gt_train.values())

# Count document frequencies across targets
token_doc_freq = Counter()
for name in df_targets["norm_name"].values:
    for t in set(name.split()):
        if len(t) >= 3 and t not in COMMON_STOP_TOKENS:
            token_doc_freq[t] += 1

print(f"Total Unique Informative Name Tokens: {len(token_doc_freq)}")
print("Top 10 highest frequency name tokens in targets:")
for t, c in token_doc_freq.most_common(10):
    print(f"  {t:<15}: {c} records")

cap_values = [10, 25, 50, 100, 250, 500, 10000]

print("\n" + "=" * 80)
print(f"{'Cap Value':<12} | {'Tokens Indexed':<15} | {'Recall':<8} | {'Retained':<9} | {'Total Cands':<11} | {'Mean/S1':<8} | {'Max/S1':<7}")
print("=" * 80)

for cap in cap_values:
    valid_tokens = {t for t, c in token_doc_freq.items() if c <= cap}
    
    # Build inverted index
    index = defaultdict(list)
    for _, r in df_targets.iterrows():
        eid = r["entity_id"]
        for t in set(r["norm_name"].split()):
            if t in valid_tokens:
                index[t].append(eid)
                
    # Query for S1
    cand_map = {}
    for _, r in df_s1.iterrows():
        s1_id = r["entity_id"]
        scores = Counter()
        for t in set(r["norm_name"].split()):
            if t in index:
                for eid in index[t]:
                    scores[eid] += 1
        cand_map[s1_id] = [eid for eid, _ in scores.most_common(50)]
        
    m = evaluate_blocking(cand_map, gt_train, n_targets)
    cap_label = str(cap) if cap < 10000 else "No Cap"
    print(f"{cap_label:<12} | {len(valid_tokens):<15} | {m['blocking_recall']*100:6.2f}% | {m['captured_true_matches']:4d}/{total_true} | {m['total_candidates']:11d} | {m['mean_candidates_per_s1']:8.1f} | {m['max_candidates_per_s1']:7d}")

print("=" * 80)

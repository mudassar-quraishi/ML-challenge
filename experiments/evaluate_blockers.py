"""Benchmark each blocking strategy independently on the ground truth.

Measures:
- Blocking recall
- Total candidate volume
- Mean, Median, P95, Max candidates per S1
- Reduction ratio
- Individual contribution and pairwise overlap
"""

import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe, strip_accents_and_normalize
from src.blocking import evaluate_blocking, COMMON_STOP_TOKENS

sys.stdout.reconfigure(encoding='utf-8')

# Address stop words that appear too frequently to be useful blocking keys
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

# Domain extensions
DOMAIN_SUFFIXES = r"\.(com|in|org|net|fr|co|edu|biz|info|io)\b"


# =====================================================================
# INDEPENDENT BLOCKER DEFINITIONS
# =====================================================================

class BlockerA_ExactName:
    name = "Blocker A (Exact Normalized Name)"
    def __init__(self):
        self.index = defaultdict(list)
    def build(self, df_targets: pd.DataFrame):
        for _, r in df_targets.iterrows():
            name = r["norm_name"]
            if name:
                self.index[name].append(r["entity_id"])
    def query(self, s1_row: dict) -> List[str]:
        name = s1_row["norm_name"]
        return list(self.index.get(name, []))


class BlockerB_RareToken:
    name = "Blocker B (Rare Name Tokens)"
    def __init__(self, max_doc_freq: int = 50, min_len: int = 3):
        self.max_doc_freq = max_doc_freq
        self.min_len = min_len
        self.index = defaultdict(list)
    def build(self, df_targets: pd.DataFrame):
        tok_counts = Counter()
        for name in df_targets["norm_name"].values:
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= self.min_len and t not in COMMON_STOP_TOKENS:
                    tok_counts[t] += 1
        valid_toks = {t for t, c in tok_counts.items() if c <= self.max_doc_freq}
        for _, r in df_targets.iterrows():
            eid = r["entity_id"]
            tokens = set(r["norm_name"].split())
            for t in tokens:
                if t in valid_toks:
                    self.index[t].append(eid)
    def query(self, s1_row: dict) -> List[str]:
        tokens = set(s1_row["norm_name"].split())
        cands = Counter()
        for t in tokens:
            if t in self.index:
                for eid in self.index[t]:
                    cands[eid] += 1
        return [eid for eid, _ in cands.most_common(50)]


class BlockerC_CharNgram:
    name = "Blocker C (Char 3-Gram TF-IDF Cosine)"
    def __init__(self, min_sim: float = 0.40, top_k: int = 30):
        self.min_sim = min_sim
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
        self.target_matrix = None
        self.target_ids = []
    def build(self, df_targets: pd.DataFrame):
        names = df_targets["norm_name"].fillna("").tolist()
        self.target_ids = df_targets["entity_id"].tolist()
        self.target_matrix = self.vectorizer.fit_transform(names)
    def query(self, s1_row: dict) -> List[str]:
        name = s1_row["norm_name"]
        if not name or len(name) < 3:
            return []
        s1_vec = self.vectorizer.transform([name])
        sims = (s1_vec * self.target_matrix.T).toarray().ravel()
        # Find indices with sim >= min_sim
        above_thresh = np.where(sims >= self.min_sim)[0]
        if len(above_thresh) == 0:
            return []
        if len(above_thresh) > self.top_k:
            top_idx = above_thresh[np.argsort(sims[above_thresh])[-self.top_k:]]
        else:
            top_idx = above_thresh
        return [self.target_ids[idx] for idx in top_idx]


class BlockerD_UrlDomain:
    name = "Blocker D (URL / Domain Stripped Name)"
    def __init__(self):
        self.clean_name_index = defaultdict(list)
    @staticmethod
    def extract_core_brand(name: str) -> str:
        text = name.lower()
        text = re.sub(r"^@", "", text)
        text = re.sub(DOMAIN_SUFFIXES, "", text)
        text = re.sub(r"^www\.", "", text)
        text = re.sub(r"[^\w]", "", text)
        return text
    def build(self, df_targets: pd.DataFrame):
        for _, r in df_targets.iterrows():
            core = self.extract_core_brand(r["business_name"])
            if len(core) >= 4:
                self.clean_name_index[core].append(r["entity_id"])
    def query(self, s1_row: dict) -> List[str]:
        core = self.extract_core_brand(s1_row["business_name"])
        if len(core) >= 4 and core in self.clean_name_index:
            return list(self.clean_name_index[core])
        return []


class BlockerE_AddressDigits:
    name = "Blocker E (Address Digits / PIN Code)"
    def __init__(self, max_doc_freq: int = 150):
        self.max_doc_freq = max_doc_freq
        self.index = defaultdict(list)
    def build(self, df_targets: pd.DataFrame):
        dig_counts = Counter()
        for _, r in df_targets.iterrows():
            cntry = r["norm_country"]
            digits = r.get("address_digits", [])
            for d in digits[:3]:
                if len(d) >= 3:
                    dig_counts[(d, cntry)] += 1
        valid_keys = {k for k, c in dig_counts.items() if c <= self.max_doc_freq}
        for _, r in df_targets.iterrows():
            eid = r["entity_id"]
            cntry = r["norm_country"]
            digits = r.get("address_digits", [])
            for d in digits[:3]:
                if (d, cntry) in valid_keys:
                    self.index[(d, cntry)].append(eid)
    def query(self, s1_row: dict) -> List[str]:
        cntry = s1_row["norm_country"]
        digits = s1_row.get("address_digits", [])
        cands = Counter()
        for d in digits[:3]:
            key = (d, cntry)
            if key in self.index:
                for eid in self.index[key]:
                    cands[eid] += 1
        return [eid for eid, _ in cands.most_common(40)]


class BlockerF_AddressTokens:
    name = "Blocker F (Informative Address Tokens)"
    def __init__(self, max_doc_freq: int = 50, min_len: int = 4):
        self.max_doc_freq = max_doc_freq
        self.min_len = min_len
        self.index = defaultdict(list)
    def build(self, df_targets: pd.DataFrame):
        tok_counts = Counter()
        for addr in df_targets["norm_address"].values:
            tokens = set(addr.split())
            for t in tokens:
                if len(t) >= self.min_len and t not in ADDRESS_STOP_TOKENS:
                    tok_counts[t] += 1
        valid_toks = {t for t, c in tok_counts.items() if c <= self.max_doc_freq}
        for _, r in df_targets.iterrows():
            eid = r["entity_id"]
            tokens = set(r["norm_address"].split())
            for t in tokens:
                if t in valid_toks:
                    self.index[t].append(eid)
    def query(self, s1_row: dict) -> List[str]:
        tokens = set(s1_row["norm_address"].split())
        cands = Counter()
        for t in tokens:
            if t in self.index:
                for eid in self.index[t]:
                    cands[eid] += 1
        # Require at least 2 shared address tokens to keep precision clean
        filtered = [eid for eid, count in cands.items() if count >= 2]
        return sorted(filtered, key=lambda x: cands[x], reverse=True)[:40]


# =====================================================================
# EXECUTION & MEASUREMENT
# =====================================================================

print("\n>>> Loading Data...")
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
print(f"Dataset Loaded: S1={len(df_s1)}, Targets={n_targets}, True Matches={total_true}")

blockers = [
    BlockerA_ExactName(),
    BlockerB_RareToken(max_doc_freq=50),
    BlockerC_CharNgram(min_sim=0.45, top_k=30),
    BlockerD_UrlDomain(),
    BlockerE_AddressDigits(max_doc_freq=150),
    BlockerF_AddressTokens(max_doc_freq=50)
]

blocker_results = {}
blocker_candidate_maps = {}

print("\n" + "=" * 80)
print(f"{'Blocker Name':<38} | {'Recall':<8} | {'Retained':<9} | {'Cands':<7} | {'Mean/S1':<7} | {'P95':<5} | {'Red. Ratio':<10}")
print("=" * 80)

for blocker in blockers:
    t0 = time.time()
    blocker.build(df_targets)
    c_map = {}
    for _, r in df_s1.iterrows():
        c_map[r["entity_id"]] = blocker.query(r.to_dict())
    elapsed = time.time() - t0
    
    metrics = evaluate_blocking(c_map, gt_train, n_targets)
    blocker_results[blocker.name] = metrics
    blocker_candidate_maps[blocker.name] = c_map
    
    print(f"{blocker.name:<38} | {metrics['blocking_recall']*100:6.2f}% | {metrics['captured_true_matches']:4d}/{total_true} | {metrics['total_candidates']:7d} | {metrics['mean_candidates_per_s1']:7.1f} | {metrics['p95_candidates_per_s1']:5.1f} | {metrics['reduction_ratio']*100:8.4f}%")

print("=" * 80)

# =====================================================================
# UNION COMBINATIONS EVALUATION
# =====================================================================

print("\n>>> EVALUATING UNION STRATEGIES...")

def evaluate_union(names_to_combine: List[str], max_cap_per_s1: int = 70) -> dict:
    union_map = {}
    for _, r in df_s1.iterrows():
        s1_id = r["entity_id"]
        combined = Counter()
        for b_name in names_to_combine:
            cands = blocker_candidate_maps[b_name].get(s1_id, [])
            for rank, cid in enumerate(cands):
                # Earlier rank gets slightly higher score
                score = 1.0 / (rank + 1.0)
                if "Exact" in b_name:
                    score += 50.0
                elif "Url" in b_name:
                    score += 20.0
                combined[cid] += score
        if not combined:
            union_map[s1_id] = []
        else:
            union_map[s1_id] = [cid for cid, _ in combined.most_common(max_cap_per_s1)]
    return evaluate_blocking(union_map, gt_train, n_targets)

unions_to_test = [
    ("Baseline (A + B + E)", [
        BlockerA_ExactName.name,
        BlockerB_RareToken.name,
        BlockerE_AddressDigits.name
    ]),
    ("Union: A + B + C (Name + CharNgram)", [
        BlockerA_ExactName.name,
        BlockerB_RareToken.name,
        BlockerC_CharNgram.name
    ]),
    ("Union: A + B + C + D (Names + CharNgram + URL)", [
        BlockerA_ExactName.name,
        BlockerB_RareToken.name,
        BlockerC_CharNgram.name,
        BlockerD_UrlDomain.name
    ]),
    ("Full Multi-Modal Union: A + B + C + D + E + F", [
        BlockerA_ExactName.name,
        BlockerB_RareToken.name,
        BlockerC_CharNgram.name,
        BlockerD_UrlDomain.name,
        BlockerE_AddressDigits.name,
        BlockerF_AddressTokens.name
    ]),
]

print("\n" + "=" * 85)
print(f"{'Union Strategy':<45} | {'Recall':<8} | {'Retained':<9} | {'Cands':<7} | {'Mean/S1':<7} | {'P95':<5}")
print("=" * 85)

for label, b_names in unions_to_test:
    m = evaluate_union(b_names, max_cap_per_s1=65)
    print(f"{label:<45} | {m['blocking_recall']*100:6.2f}% | {m['captured_true_matches']:4d}/{total_true} | {m['total_candidates']:7d} | {m['mean_candidates_per_s1']:7.1f} | {m['p95_candidates_per_s1']:5.1f}")

print("=" * 85)

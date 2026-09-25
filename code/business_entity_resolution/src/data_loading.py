"""Data loading utilities for Business Entity Resolution.

Memory-safe, chunked and streaming data loaders with support for:
- Local smoke testing with deterministic entity sampling
- Full-scale streaming and chunked processing for Google Colab
"""

import os
import random
from typing import Dict, List, Set, Tuple
import pandas as pd
from .config import PipelineConfig


def read_source_tsv(path: str, nrows: int = None) -> pd.DataFrame:
    """Read a tab-separated source file ensuring strict string types and no NaN conversion."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
        encoding="utf-8"
    )


def load_ground_truth_dict(gt_path: str, target_s1_ids: Set[str] = None) -> Dict[str, Set[str]]:
    """Stream ground truth file and return mapping: source1_entity_id -> set of matched_entity_ids."""
    gt_map: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            if target_s1_ids is not None and s1_id not in target_s1_ids:
                continue
            if len(parts) > 1 and parts[1].strip():
                matches = set(parts[1].strip().split(","))
            else:
                matches = set()
            gt_map[s1_id] = matches
    return gt_map


def load_train_data(config: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Load training data (S1, S2, S3, ground truth).
    
    If config.LOCAL_SMOKE_TEST is True, loads a coherent micro-universe:
    - samples config.SMOKE_SAMPLE_SIZE S1 records
    - includes all their true S2/S3 matches
    - includes a pool of distractor S2/S3 records
    """
    s1_path = os.path.join(config.TRAIN_DIR, "train_source1.tsv")
    s2_path = os.path.join(config.TRAIN_DIR, "train_source2.tsv")
    s3_path = os.path.join(config.TRAIN_DIR, "train_source3.tsv")
    gt_path = os.path.join(config.TRAIN_DIR, "train_ground_truth.tsv")

    if not config.LOCAL_SMOKE_TEST:
        print("[DataLoading] Loading FULL training dataset...")
        df_s1 = read_source_tsv(s1_path)
        df_s2 = read_source_tsv(s2_path)
        df_s3 = read_source_tsv(s3_path)
        gt_map = load_ground_truth_dict(gt_path)
        print(f"[DataLoading] Full train loaded: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}, GT={len(gt_map)}")
        return df_s1, df_s2, df_s3, gt_map

    # Smoke test mode
    print(f"[DataLoading] Smoke test enabled: Sampling {config.SMOKE_SAMPLE_SIZE} S1 entities...")
    df_s1 = read_source_tsv(s1_path, nrows=config.SMOKE_SAMPLE_SIZE)
    s1_ids = set(df_s1["entity_id"].values)
    gt_map = load_ground_truth_dict(gt_path, target_s1_ids=s1_ids)
    
    # Collect all needed matching IDs
    needed_s2_s3_ids = set()
    for matches in gt_map.values():
        needed_s2_s3_ids.update(matches)

    print(f"[DataLoading] Smoke test: {len(needed_s2_s3_ids)} true target matches identified in GT.")

    # Stream S2 and S3 to get true matches + distractors
    def sample_target_source(path: str, needed_ids: Set[str], max_distractors: int) -> pd.DataFrame:
        matched_rows = []
        distractor_rows = []
        random.seed(config.RANDOM_SEED)
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if not parts or not parts[0]:
                    continue
                row_dict = {
                    "entity_id": parts[0],
                    "business_name": parts[1] if len(parts) > 1 else "",
                    "business_address": parts[2] if len(parts) > 2 else "",
                    "country": parts[3] if len(parts) > 3 else ""
                }
                if row_dict["entity_id"] in needed_ids:
                    matched_rows.append(row_dict)
                elif len(distractor_rows) < max_distractors:
                    distractor_rows.append(row_dict)
                elif random.random() < 0.005 and len(distractor_rows) < max_distractors * 2:
                    distractor_rows.append(row_dict)
                    
        combined = matched_rows + distractor_rows[:max_distractors]
        return pd.DataFrame(combined)

    distractor_cap = config.SMOKE_SAMPLE_SIZE * 3
    df_s2 = sample_target_source(s2_path, needed_s2_s3_ids, distractor_cap)
    df_s3 = sample_target_source(s3_path, needed_s2_s3_ids, distractor_cap)

    print(f"[DataLoading] Smoke sample ready: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}, GT={len(gt_map)}")
    return df_s1, df_s2, df_s3, gt_map


def load_test_data(config: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load test data (S1, S2, S3).
    
    If config.LOCAL_SMOKE_TEST is True, loads a small slice for fast local verification.
    """
    s1_path = os.path.join(config.TEST_DIR, "test_source1.tsv")
    s2_path = os.path.join(config.TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(config.TEST_DIR, "test_source3.tsv")

    if not config.LOCAL_SMOKE_TEST:
        print("[DataLoading] Loading FULL test dataset...")
        df_s1 = read_source_tsv(s1_path)
        df_s2 = read_source_tsv(s2_path)
        df_s3 = read_source_tsv(s3_path)
        print(f"[DataLoading] Full test loaded: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}")
        return df_s1, df_s2, df_s3

    print(f"[DataLoading] Smoke test enabled for TEST split: {config.SMOKE_SAMPLE_SIZE} S1 entities...")
    df_s1 = read_source_tsv(s1_path, nrows=config.SMOKE_SAMPLE_SIZE)
    df_s2 = read_source_tsv(s2_path, nrows=config.SMOKE_SAMPLE_SIZE * 3)
    df_s3 = read_source_tsv(s3_path, nrows=config.SMOKE_SAMPLE_SIZE * 3)
    print(f"[DataLoading] Smoke test loaded: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}")
    return df_s1, df_s2, df_s3

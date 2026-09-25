"""Model training and validation module for Business Entity Resolution.

Trains a supervised pairwise matching model (LightGBM) using engineered candidate features
and calibrates the decision threshold using macro F0.5.
"""

import json
import os
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .config import PipelineConfig
from .evaluate import (
    calibrate_threshold,
    entity_level_train_val_split,
    evaluate_predictions
)
from .features import FEATURE_NAMES, build_feature_matrix


def prepare_training_pairs(
    candidates_map: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    s1_ids: Set[str]
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Prepare candidate pairs and binary labels for a set of S1 entities."""
    pairs = []
    labels = []
    
    for s1_id in s1_ids:
        cands = candidates_map.get(s1_id, [])
        gt_matches = ground_truth.get(s1_id, set())
        for cid in cands:
            pairs.append((s1_id, cid))
            labels.append(1 if cid in gt_matches else 0)
            
    return pairs, np.array(labels, dtype=np.int32)


def train_matching_model(
    candidates_map: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    s1_dict: Dict[str, dict],
    target_dict: Dict[str, dict],
    config: PipelineConfig
) -> Tuple[LGBMClassifier, float, dict]:
    """Train the matching model, perform entity-level validation, and calibrate threshold."""
    all_s1_ids = list(candidates_map.keys())
    train_s1, val_s1 = entity_level_train_val_split(
        all_s1_ids,
        val_ratio=config.VALIDATION_SPLIT_RATIO,
        seed=config.RANDOM_SEED
    )
    print(f"[Train] Entity split: {len(train_s1)} Train S1, {len(val_s1)} Val S1 (Zero leakage).")
    
    # 1. Build training features
    train_pairs, y_train = prepare_training_pairs(candidates_map, ground_truth, train_s1)
    print(f"[Train] Training pairs: {len(train_pairs)} (Positives: {np.sum(y_train == 1)}, Negatives: {np.sum(y_train == 0)})")
    
    X_train = build_feature_matrix(train_pairs, s1_dict, target_dict)
    
    # 2. Build validation features
    val_pairs, y_val = prepare_training_pairs(candidates_map, ground_truth, val_s1)
    print(f"[Train] Validation pairs: {len(val_pairs)} (Positives: {np.sum(y_val == 1)}, Negatives: {np.sum(y_val == 0)})")
    
    X_val = build_feature_matrix(val_pairs, s1_dict, target_dict)
    
    # 3. Fit LightGBM classifier
    n_estimators = 50 if config.LOCAL_SMOKE_TEST else 200
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.08,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_SEED,
        n_jobs=config.N_JOBS,
        importance_type="gain",
        verbose=-1
    )
    print("[Train] Fitting LightGBM model...")
    model.fit(X_train, y_train)
    
    # Feature importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("[Train] Top 5 Feature Importances (gain):")
    for fname, imp in feat_imp[:5]:
        print(f"  - {fname}: {imp:.4f}")
        
    # 4. Predict on validation candidate pairs
    val_probs = model.predict_proba(X_val)[:, 1] if len(val_pairs) > 0 else np.array([])
    
    # Group validation scores by S1 entity
    val_candidate_scores: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in val_s1}
    for (s1_id, cid), prob in zip(val_pairs, val_probs):
        val_candidate_scores[s1_id].append((cid, float(prob)))
        
    # Validation ground truth subset
    val_gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1}
    
    # 5. Calibrate threshold using competition macro F0.5
    best_thresh, best_metrics, _ = calibrate_threshold(val_candidate_scores, val_gt)
    
    # 6. Save model and metadata
    model_path = os.path.join(config.MODEL_DIR, "matcher_lgbm.joblib")
    meta_path = os.path.join(config.MODEL_DIR, "calibration_meta.json")
    joblib.dump(model, model_path)
    
    meta = {
        "best_threshold": float(best_thresh),
        "validation_metrics": best_metrics,
        "feature_names": FEATURE_NAMES,
        "feature_importances": {k: float(v) for k, v in feat_imp}
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[Train] Model saved to {model_path}")
    print(f"[Train] Calibration saved to {meta_path}")
    return model, best_thresh, best_metrics

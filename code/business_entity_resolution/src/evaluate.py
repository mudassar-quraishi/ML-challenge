"""Evaluation and threshold calibration for Business Entity Resolution.

Implements the exact official competition metric:
- Macro-averaged F0.5 over all Source 1 entities (including singletons)
- Entity-level train/validation splitting
- Threshold sweep and calibration
"""

import random
from typing import Dict, List, Set, Tuple
import numpy as np
from .config import PipelineConfig


def calculate_entity_f05(gt_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """Calculate F0.5, Precision, and Recall for a single S1 entity according to official competition rules."""
    # Singleton case: No true matches in ground truth
    if len(gt_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0  # Correctly predicted empty
        else:
            return 0.0, 0.0, 0.0  # False merge on singleton

    # Non-singleton case: Has true matches, but prediction is empty
    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0  # Missed all matches

    # Non-empty prediction and non-empty ground truth
    tp = len(gt_set.intersection(pred_set))
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    
    # Official F0.5 formula: (1.25 * P * R) / (0.25 * P + R)
    denominator = (0.25 * precision + recall)
    f05 = (1.25 * precision * recall) / denominator if denominator > 0 else 0.0
    
    return float(f05), float(precision), float(recall)


def evaluate_predictions(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]]
) -> dict:
    """Calculate macro-averaged F0.5 across all evaluated S1 entities."""
    f05_scores = []
    precision_scores = []
    recall_scores = []
    
    singleton_total = 0
    singleton_correct = 0
    non_singleton_total = 0
    non_singleton_f05 = []
    
    for s1_id, gt_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())
        f05, prec, rec = calculate_entity_f05(gt_set, pred_set)
        
        f05_scores.append(f05)
        precision_scores.append(prec)
        recall_scores.append(rec)
        
        if len(gt_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_total += 1
            non_singleton_f05.append(f05)
            
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_prec = float(np.mean(precision_scores)) if precision_scores else 0.0
    macro_rec = float(np.mean(recall_scores)) if recall_scores else 0.0
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 0.0
    non_singleton_mean_f05 = float(np.mean(non_singleton_f05)) if non_singleton_f05 else 0.0
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "singleton_total": singleton_total,
        "singleton_accuracy": float(singleton_acc),
        "non_singleton_total": non_singleton_total,
        "non_singleton_macro_f05": non_singleton_mean_f05,
        "total_evaluated_s1": len(f05_scores)
    }


def calibrate_threshold(
    val_candidate_scores: Dict[str, List[Tuple[str, float]]],
    val_ground_truth: Dict[str, Set[str]],
    threshold_grid: List[float] = None
) -> Tuple[float, dict, List[dict]]:
    """Perform sweep over threshold grid to find optimal F0.5 cut-off."""
    if threshold_grid is None:
        threshold_grid = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]
        
    best_thresh = 0.50
    best_metrics = {}
    best_f05 = -1.0
    all_results = []
    
    for thresh in threshold_grid:
        preds: Dict[str, Set[str]] = {}
        for s1_id in val_ground_truth.keys():
            cand_pairs = val_candidate_scores.get(s1_id, [])
            matched = {cid for cid, score in cand_pairs if score >= thresh}
            preds[s1_id] = matched
            
        metrics = evaluate_predictions(preds, val_ground_truth)
        metrics["threshold"] = thresh
        all_results.append(metrics)
        
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_thresh = thresh
            best_metrics = metrics
            
    print(f"[Calibration] Best Threshold: {best_thresh:.2f} | Macro F0.5: {best_f05:.4f} | Prec: {best_metrics['macro_precision']:.4f} | Rec: {best_metrics['macro_recall']:.4f}")
    return best_thresh, best_metrics, all_results


def entity_level_train_val_split(
    s1_ids: List[str],
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """Deterministically split S1 entity IDs into train and validation sets with zero leakage."""
    rng = random.Random(seed)
    shuffled = list(s1_ids)
    rng.shuffle(shuffled)
    
    n_val = int(len(shuffled) * val_ratio)
    val_s1 = set(shuffled[:n_val])
    train_s1 = set(shuffled[n_val:])
    
    # Assert zero overlap
    assert len(train_s1.intersection(val_s1)) == 0, "Data leakage detected: train and val S1 sets overlap!"
    return train_s1, val_s1

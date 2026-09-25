"""Main orchestrator for the Business Entity Resolution pipeline.

Connects data loading, normalization, blocking, feature engineering,
model training, threshold calibration, inference, output generation, and validation.
"""

import os
import subprocess
import sys
import time
from typing import Optional
import pandas as pd

from .config import PipelineConfig
from .data_loading import load_train_data, load_test_data
from .normalization import normalize_dataframe
from .blocking import CandidateBlocker, evaluate_blocking
from .train import train_matching_model
from .predict import run_batch_inference
from .make_submission import write_submission_files


def run_full_pipeline(config: PipelineConfig) -> dict:
    """Execute the end-to-end entity resolution workflow."""
    start_time = time.time()
    print("=" * 70)
    print(" Amazon ML Challenge 2026 - Business Entity Resolution Pipeline")
    print("=" * 70)
    print(f"Data Root:        {config.DATA_ROOT}")
    print(f"Train Dir:        {config.TRAIN_DIR}")
    print(f"Test Dir:         {config.TEST_DIR}")
    print(f"Output Dir:       {config.OUTPUT_DIR}")
    print(f"Smoke Test Mode:  {config.LOCAL_SMOKE_TEST}")
    if config.LOCAL_SMOKE_TEST:
        print(f"Smoke Sample:     {config.SMOKE_SAMPLE_SIZE} S1 entities")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # Step 1: Load and normalize training data
    # -------------------------------------------------------------
    print("\n>>> STEP 1: Loading & Normalizing Training Data...")
    df_s1_train, df_s2_train, df_s3_train, gt_train = load_train_data(config)
    
    df_s1_train = normalize_dataframe(df_s1_train)
    df_s2_train = normalize_dataframe(df_s2_train)
    df_s3_train = normalize_dataframe(df_s3_train)
    
    # Combined target pool for training
    df_targets_train = pd.concat([df_s2_train, df_s3_train], ignore_index=True)
    
    # -------------------------------------------------------------
    # Step 2: Training Blocking & Evaluation
    # -------------------------------------------------------------
    print("\n>>> STEP 2: Candidate Blocking on Training Data...")
    train_blocker = CandidateBlocker(config)
    train_blocker.build_index(df_targets_train)
    train_candidates = train_blocker.generate_candidates(df_s1_train)
    
    blocking_metrics = evaluate_blocking(
        train_candidates,
        gt_train,
        n_targets=len(df_targets_train)
    )
    print("\n--- Blocking Evaluation Results ---")
    print(f"Blocking Recall:       {blocking_metrics['blocking_recall']*100:.2f}%")
    print(f"Total True Matches:    {blocking_metrics['total_true_matches']}")
    print(f"Captured True Matches: {blocking_metrics['captured_true_matches']}")
    print(f"Total Candidates:      {blocking_metrics['total_candidates']}")
    print(f"Mean Cands / S1:       {blocking_metrics['mean_candidates_per_s1']:.1f}")
    print(f"P95 Cands / S1:        {blocking_metrics['p95_candidates_per_s1']:.1f}")
    print(f"Reduction Ratio:       {blocking_metrics['reduction_ratio']*100:.4f}%")
    
    # Prepare metadata lookups for feature generation
    s1_train_dict = {
        row["entity_id"]: row.to_dict() for _, row in df_s1_train.iterrows()
    }
    target_train_dict = train_blocker.target_records
    
    # -------------------------------------------------------------
    # Step 3: Train Model & Calibrate Threshold
    # -------------------------------------------------------------
    print("\n>>> STEP 3: Training Supervised Matcher & Calibrating Threshold...")
    model, calibrated_threshold, val_metrics = train_matching_model(
        train_candidates,
        gt_train,
        s1_train_dict,
        target_train_dict,
        config
    )
    
    # -------------------------------------------------------------
    # Step 4: Test Data Loading & Normalization
    # -------------------------------------------------------------
    print("\n>>> STEP 4: Processing Test Data...")
    df_s1_test, df_s2_test, df_s3_test = load_test_data(config)
    df_s1_test = normalize_dataframe(df_s1_test)
    df_s2_test = normalize_dataframe(df_s2_test)
    df_s3_test = normalize_dataframe(df_s3_test)
    
    df_targets_test = pd.concat([df_s2_test, df_s3_test], ignore_index=True)
    
    # -------------------------------------------------------------
    # Step 5: Test Blocking
    # -------------------------------------------------------------
    print("\n>>> STEP 5: Blocking on Test Data...")
    test_blocker = CandidateBlocker(config)
    test_blocker.build_index(df_targets_test)
    test_candidates = test_blocker.generate_candidates(df_s1_test)
    
    s1_test_dict = {
        row["entity_id"]: row.to_dict() for _, row in df_s1_test.iterrows()
    }
    target_test_dict = test_blocker.target_records
    
    # -------------------------------------------------------------
    # Step 6: Batch Inference
    # -------------------------------------------------------------
    print("\n>>> STEP 6: Running Batch Inference on Test Set...")
    final_candidates, final_matches = run_batch_inference(
        model=model,
        candidates_map=test_candidates,
        s1_dict=s1_test_dict,
        target_dict=target_test_dict,
        threshold=calibrated_threshold,
        batch_size=500 if config.LOCAL_SMOKE_TEST else 2000
    )
    
    # -------------------------------------------------------------
    # Step 7: Output Submission Generation
    # -------------------------------------------------------------
    print("\n>>> STEP 7: Writing Submission TSVs...")
    required_test_s1_ids = list(df_s1_test["entity_id"].values)
    matching_path, candidate_path = write_submission_files(
        required_s1_ids=required_test_s1_ids,
        candidates_map=final_candidates,
        matches_map=final_matches,
        output_dir=config.OUTPUT_DIR
    )
    
    # -------------------------------------------------------------
    # Step 8: Validate Outputs with Official Validator
    # -------------------------------------------------------------
    print("\n>>> STEP 8: Running Official Submission Validator...")
    validator_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "utils", "validate_submission.py"
    )
    if not os.path.exists(validator_script):
        validator_script = "utils/validate_submission.py"
        
    validation_test_dir = config.TEST_DIR
    if config.LOCAL_SMOKE_TEST:
        smoke_dir = os.path.join(config.CACHE_DIR, "smoke_test_eval")
        os.makedirs(smoke_dir, exist_ok=True)
        # Write minimal source files for validator to verify against
        cols = ["entity_id", "business_name", "business_address", "country"]
        df_s1_test[cols].to_csv(os.path.join(smoke_dir, "test_source1.tsv"), sep="\t", index=False)
        df_s2_test[cols].to_csv(os.path.join(smoke_dir, "test_source2.tsv"), sep="\t", index=False)
        df_s3_test[cols].to_csv(os.path.join(smoke_dir, "test_source3.tsv"), sep="\t", index=False)
        validation_test_dir = smoke_dir

    validation_status = "UNKNOWN"
    if os.path.exists(validator_script):
        cmd = [
            sys.executable,
            validator_script,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", validation_test_dir
        ]
        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("Validator Output:\n" + res.stdout)
        if res.stderr:
            print("Validator Errors:\n" + res.stderr)
        validation_status = "PASS" if res.returncode == 0 else "FAIL"
    else:
        print(f"[Warning] Validator script not found at {validator_script}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" Pipeline Finished in {elapsed:.1f}s | Validator Status: {validation_status}")
    print("=" * 70)
    
    return {
        "elapsed_seconds": elapsed,
        "blocking_metrics": blocking_metrics,
        "calibrated_threshold": calibrated_threshold,
        "validation_metrics": val_metrics,
        "validation_status": validation_status,
        "matching_path": matching_path,
        "candidate_path": candidate_path
    }

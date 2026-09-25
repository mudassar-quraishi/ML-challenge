#!/usr/bin/env python3
"""Google Colab & CLI Execution Entry Point for Business Entity Resolution Pipeline.

Usage in Google Colab or terminal:

1. Local smoke test (1000 entities, fast, low memory):
   python colab/run_pipeline.py --smoke-test

2. Full dataset execution (in Google Colab or high-RAM machine):
   python colab/run_pipeline.py --data-root dataset --output-dir output

3. Custom paths or Google Drive mount:
   python colab/run_pipeline.py --data-root /content/drive/MyDrive/AmazonML/dataset --output-dir /content/drive/MyDrive/AmazonML/output
"""

import argparse
import os
import sys

# Ensure code/business_entity_resolution is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.pipeline import run_full_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Business Entity Resolution Pipeline (Amazon ML Challenge 2026)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test on small deterministic sample (for 8 GB RAM local machines)"
    )
    parser.add_argument(
        "--smoke-sample-size",
        type=int,
        default=1000,
        help="Number of S1 entities to sample during smoke test (default: 1000)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Path to dataset root folder containing train/ and test/ (default: dataset)"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="",
        help="Path to training directory (default: <data-root>/train)"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="",
        help="Path to test directory (default: <data-root>/test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save matching_results.tsv and candidate_pairs.tsv (default: output)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save model weights and calibration metadata (default: models)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for cache files (default: cache)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU cores for training and scoring (-1 for all cores)"
    )
    parser.add_argument(
        "--blocking-mode",
        type=str,
        default="multimodal",
        choices=["baseline", "multimodal"],
        help="Blocking strategy: 'baseline' (Phase 1) or 'multimodal' (Phase 3 multi-modal union)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = PipelineConfig(
        DATA_ROOT=args.data_root,
        TRAIN_DIR=args.train_dir,
        TEST_DIR=args.test_dir,
        OUTPUT_DIR=args.output_dir,
        MODEL_DIR=args.model_dir,
        CACHE_DIR=args.cache_dir,
        LOCAL_SMOKE_TEST=args.smoke_test,
        SMOKE_SAMPLE_SIZE=args.smoke_sample_size,
        BLOCKING_MODE=args.blocking_mode,
        N_JOBS=args.n_jobs
    )
    
    results = run_full_pipeline(config)
    print("\nExecution Summary:")
    print(f"Status: {results['validation_status']}")
    print(f"Elapsed Time: {results['elapsed_seconds']:.2f} seconds")
    return 0 if results["validation_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

Production-quality, submission-ready entity resolution solution designed for multi-source business record linkage.

## System Architecture

```
[Raw TSVs] ──> [Memory-Safe Data Loader] ──> [Multilingual Normalization]
                                                        │
┌───────────────────────────────────────────────────────┘
▼
[High-Recall Candidate Blocking] ──> Candidate Pairs
                                            │
┌───────────────────────────────────────────┘
▼
[Pairwise Feature Engineering] (String/Token/Digit/Country/Interactions)
      │
      ├──> [Entity-Level Train/Val Split] (Zero Leakage)
      │          │
      │          ▼
      ├──> [Supervised LightGBM Matcher]
      │          │
      │          ▼
      └──> [F0.5 Threshold Calibration] (Macro-averaged over S1 including singletons)
                 │
                 ▼
      [Batch Test Inference] ──> [Submission TSV Formatter] ──> [validate_submission.py]
```

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Configuration system (env vars + CLI)
│   ├── data_loading.py       # Memory-safe chunked & streaming loaders
│   ├── normalization.py      # Unicode, legal suffix & address standardization
│   ├── blocking.py           # Inverted token & digit indexes, candidate caps
│   ├── features.py           # 22 pairwise string, token, digit & country features
│   ├── evaluate.py           # Exact competition macro F0.5 & threshold sweep
│   ├── train.py              # Supervised LightGBM training & validation
│   ├── predict.py            # Memory-efficient batch candidate scoring
│   ├── make_submission.py    # Formatter for matching_results.tsv & candidate_pairs.tsv
│   └── pipeline.py           # End-to-end pipeline orchestrator
├── README.md                 # This file
└── requirements.txt          # Pinned package versions
```

## Quickstart

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Run Local Smoke Test (Fast, 8 GB RAM Friendly)

```bash
python ../../colab/run_pipeline.py --smoke-test --smoke-sample-size 1000
```

### 3. Run Full-Scale Pipeline (Google Colab / Cloud Server)

```bash
python ../../colab/run_pipeline.py --data-root dataset --output-dir output
```

## Core Pipeline Modules

- **`config.py`**: Manages paths (`DATA_ROOT`, `TRAIN_DIR`, `TEST_DIR`, `OUTPUT_DIR`, `MODEL_DIR`), execution flags (`LOCAL_SMOKE_TEST`), and hyper-parameters.
- **`data_loading.py`**: Reads `.tsv` files with explicit `sep="\t"` and string typing. In smoke mode, extracts a coherent micro-universe preserving true positive pairs.
- **`normalization.py`**: Decomposes accents (NFKD) for multilingual robustness (French, Hindi, English), maps legal suffixes, and extracts address digits.
- **`blocking.py`**: Generates candidates using a union of exact names, informative token inverted lists, and address digit keys, enforcing a controlled candidate cap.
- **`features.py`**: Computes 22 deterministic features including RapidFuzz ratios, token Jaccard, digit overlaps, open-world country equality, and cross-feature interactions.
- **`evaluate.py`**: Implements the official macro F0.5 evaluation metric:
  $$\text{F}_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Singletons score 1.0 if predicted empty, 0.0 otherwise.
- **`train.py`**: Fits a LightGBM classifier with entity-level splitting (no S1 leakage) and calibrates decision threshold.
- **`predict.py`**: Performs batched test inference without holding large dense matrices in memory.
- **`make_submission.py`**: Emits `output/matching_results.tsv` and `output/candidate_pairs.tsv`, guaranteeing exact schema compliance.

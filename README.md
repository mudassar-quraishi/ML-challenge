# Amazon ML Challenge 2026 — Business Entity Resolution

A modular, production-grade Entity Resolution system resolving noisy multi-source business records across Source 1 (reference), Source 2, and Source 3.

---

## Hardware Constraint & Dual-Environment Architecture

| Environment | Purpose | Hardware Target | Execution Command |
|---|---|---|---|
| **Local Machine** | Code development, smoke testing, pipeline validation | AMD Ryzen 3 5000-series, 8 GB RAM | `python colab/run_pipeline.py --smoke-test` |
| **Google Colab** | Full-scale training, candidate generation & inference | High-RAM GPU/CPU Runtime (12–25 GB RAM) | `python colab/run_pipeline.py --data-root <path>` |

- **No Full Cartesian Product**: On 11.7M records, $S_1 \times (S_2 + S_3) \approx 17.2 \times 10^{12}$ pairs. Blocking and inverted indices reduce search space by >99.9%.
- **Memory Safety**: Uses chunked reading, inverted lists, batch feature generation, and streaming output writing.
- **Open-World Country Robustness**: Handles `US`, `India`, and test-only `France` seamlessly without hardcoded country filters.

---

## Repository Layout

```
├── dataset/                         # Dataset directory (train/ and test/)
│   ├── train/
│   │   ├── train_source1.tsv        # 2,206,821 records
│   │   ├── train_source2.tsv        # 5,034,616 records
│   │   ├── train_source3.tsv        # 5,285,603 records
│   │   └── train_ground_truth.tsv   # 2,206,821 records (7,638,365 match pairs)
│   └── test/
│       ├── test_source1.tsv         # 1,732,544 records
│       ├── test_source2.tsv         # 4,887,273 records
│       └── test_source3.tsv         # 5,082,316 records
├── utils/
│   └── validate_submission.py       # Official submission validator
├── output/
│   ├── matching_results.tsv         # Final predicted matches (leaderboard submission)
│   └── candidate_pairs.tsv          # Candidate pairs fed to inference model
├── code/
│   └── business_entity_resolution/
│       ├── src/                     # Modular pipeline source code
│       │   ├── config.py
│       │   ├── data_loading.py
│       │   ├── normalization.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   ├── evaluate.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── make_submission.py
│       │   └── pipeline.py
│       ├── README.md
│       └── requirements.txt
├── colab/
│   └── run_pipeline.py              # CLI & Colab unified execution entry point
├── Documentation_template.md        # Official methodology report
└── README.md
```

---

## 1. Local Smoke Testing (AMD Ryzen 3 / 8 GB RAM)

Run end-to-end verification in under 45 seconds using 1,000 sampled entities:

```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run smoke test
python colab/run_pipeline.py --smoke-test --smoke-sample-size 1000

# 3. Validate generated outputs
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 2. Google Colab Execution Guide (Full Dataset)

### Step 1: Open Colab & Clone Repository
In a Google Colab notebook cell:
```python
# Clone your repository
!git clone https://github.com/<your-username>/amazon-ml-challenge-2026.git
%cd amazon-ml-challenge-2026

# Install pinned dependencies
!pip install -r code/business_entity_resolution/requirements.txt
```

### Step 2: Upload / Mount Dataset
If dataset is stored on Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = "/content/drive/MyDrive/AmazonML2026/dataset"
```

### Step 3: Run Full Pipeline
```bash
!python colab/run_pipeline.py \
    --data-root "$DATA_PATH" \
    --output-dir output \
    --model-dir models \
    --n-jobs -1
```

### Step 4: Validate and Package Submission
```bash
# Run validator
!python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir "$DATA_PATH/test"

# Create final submission zip package
!zip -r submission.zip output/ code/ Documentation_template.md
```

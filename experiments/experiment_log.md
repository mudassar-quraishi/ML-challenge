# Experiment Tracking Log — Amazon ML Challenge 2026

## Methodology & Metric Guidelines
- **Primary Blocking Metric**: Blocking Recall on Ground Truth ($\frac{\text{Captured True Matches}}{\text{Total True Matches}}$)
- **Primary Optimization Metric**: Competition Macro $F_{0.5}$ over Source 1 entities (including singletons)
- **Efficiency Constraints**: Mean Candidates/S1, P95 Candidates/S1, Memory Footprint, Reduction Ratio

---

## Experiment Table

| Exp ID | Date | Blocking Strategies | Blocking Recall | Total Candidates | Mean Cands/S1 | P95 Cands/S1 | Reduction Ratio | Val Macro F0.5 | Val Precision | Val Recall | Threshold | Memory (Peak) | Runtime | Notes |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **EXP-001** *(Phase 1 Baseline)* | 2026-09-25 | Exact Name + Inverted Tokens (freq $\le$ 50) + Address Digits (freq $\le$ 200) | **91.68%** | 10,107 | 20.2 | 51.0 | 99.58% | **0.9514** | 0.9730 | 0.9127 | 0.40 | <350 MB | 17.39s | Initial working skeleton baseline on 500 S1 smoke test. Zero S1 leakage. Passed official validator. |
| **EXP-002** *(Phase 2 Union)* | 2026-09-25 | Full Multi-Modal Union: A + B + C + D + E + F | **98.93%** | 11,430 | 22.9 | 52.0 | 99.52% | **0.9815** | 0.9867 | 0.9667 | 0.30 | <380 MB | 21.40s | Recovered 129 out of 148 lost matches (+7.25% recall). Solved transliteration & URL drift. Val F0.5 jumped from 0.9514 to 0.9815. |
| **EXP-003** *(Phase 3 Production 500 S1)* | 2026-09-25 | MultiModal Production (A+B+C+D+E+F + Legal Prefix + Compound 2-Digits) | **99.16%** | 11,648 | 23.3 | 58.1 | 99.51% | **0.9805** | 0.9867 | 0.9625 | 0.30 | <380 MB | 21.17s | Dual-mode pipeline integrated into src/blocking.py. Legal prefix & 2-digit compound keys recovered 4 more matches (1,763/1,778). Validator PASS. |
| **EXP-004** *(Phase 3 Scaling 5,000 S1)* | 2026-09-25 | MultiModal Production on 5,000 S1 entities (47,362 targets) | **95.63%** | 224,181 | 44.8 | 65.0 | 99.91% | **0.9622** | 0.9764 | 0.9354 | 0.55 | <750 MB | 59.15s | End-to-end training & validation on 5,000 S1 sample. 16,604/17,362 true matches retained. High precision (0.9764) and F0.5 (0.9622). Validator PASS. |
| **EXP-005** *(Phase 3 Scaling 10,000 S1)* | 2026-09-25 | MultiModal Production on 10,000 S1 entities (94,752 targets) | **93.54%** | 518,731 | 51.9 | 65.0 | 99.95% | — | — | — | — | 598.6 MB | 358.0s | Scaling stress-test on 10,000 S1 and 94,752 targets. 32,506/34,752 true matches captured. 99.95% reduction ratio. Memory strictly controlled at 598 MB. |
| **EXP-006** *(Phase 3.5 Scaling Diagnosis)* | 2026-09-26 | Empirical diagnosis of candidate pruning vs retrieval failure on 5k and 10k S1 | **97.57% (Uncapped)** / **94.81% (Tiered 65)** / 93.57% (Flat 65) | 518,718 | 51.9 | 65.0 | 99.95% | — | — | — | — | 1,105 MB | 636.6s | Verified that 62.3% of lost matches at 10k S1 are caused by cap truncation, not blocker failure. Tiered retention boosts recall to 94.81% at cap 65 without volume increase. |

---

## Detailed Experiment History

### EXP-002: Phase 2 Multi-Modal Union Blocker
- **Commit / Tag**: Pending Phase 2 merge
- **Configuration**:
  - Sample: 500 S1 entities, 4,778 target pool (2,355 S2 + 2,423 S3)
  - True Ground Truth matches: 1,778
  - Captured matches: 1,759 (**98.93%**)
  - Blocked-out matches (Lost): 19 (**1.07%**) (down from 148 in baseline)
- **Blockers Combined**:
  - Blocker A (Exact Normalized Name): 27.56% recall, 492 cands
  - Blocker B (Rare Name Tokens, doc freq $\le$ 50): 84.14% recall, 9,267 cands
  - Blocker C (Char 3-Gram TF-IDF Cosine $\ge$ 0.45, top 30): 89.09% recall, 4,320 cands
  - Blocker D (URL / Domain Stripped Brand): 22.83% recall, 408 cands
  - Blocker E (Address Digits / PIN): 54.61% recall, 1,519 cands
  - Blocker F (Informative Address Tokens, overlap $\ge$ 2): 80.37% recall, 2,265 cands
- **Candidate Volume**: 11,430 total pairs (mean: 22.9 / S1, P95: 52.0, max: 65)
- **Model**: LightGBM (22 pairwise features, 50 trees, max_depth=6)
- **Validation Results**:
  - Threshold: 0.30
  - Macro F0.5: **0.9815** (+0.0301 over baseline)
  - Precision: **0.9867** (+0.0137 over baseline)
  - Recall: **0.9667** (+0.0540 over baseline)
  - Singleton accuracy: 100% (25/25 validation singletons correctly predicted empty)
  - Non-singleton Macro F0.5: 0.9754
- **Failure Analysis of Remaining 19 Lost Matches**:
  - 18 matches were multi-script transliteration where the entity had no street name, only a single 1- or 2-digit unit/flat number (`Flat No. 8`, `Unit 14`, `42`) which fell below the minimum digit length ($\ge 3$).
  - 1 match had severe typo in the sole word (`Keystone` vs `Keytoen`).

### EXP-001: Phase 1 Working Skeleton Baseline
- **Commit / Tag**: `phase1-baseline`
- **Configuration**:
  - Sample: 500 S1 entities, 4,778 target pool (2,355 S2 + 2,423 S3)
  - True Ground Truth matches: 1,778
  - Captured matches: 1,630
  - Blocked-out matches (Lost): 148 (8.32%)
- **Model**: LightGBM (50 trees, max_depth=6, num_leaves=31)
- **Features**: 22 pairwise string, token, digit, country & interaction features
- **Top Feature Gains**: `address_token_jaccard` (34,200), `name_ratio_x_address_jaccard` (6,871), `name_ratio_x_country` (1,236), `name_fuzz_wratio` (1,105)
- **Validation Results**:
  - Threshold: 0.40
  - Macro F0.5: 0.9514
  - Precision: 0.9730
  - Recall: 0.9127
- **Validator Status**: `PASS`

### EXP-006: Phase 3.5 Scaling Diagnosis & Candidate Pruning Audit
- **Commit / Tag**: `phase3-baseline`
- **Objective**: Determine whether recall decline across 500 S1 (99.16%) -> 5,000 S1 (95.63%) -> 10,000 S1 (93.54%) is driven by retrieval failure or candidate pruning.
- **Empirical Findings**:
  - **Raw Uncapped Blocker Recall**:
    - 500 S1: **99.16%** (1,763 / 1,778)
    - 5,000 S1: **98.15%** (17,040 / 17,362)
    - 10,000 S1: **97.57%** (33,909 / 34,752)
  - **Root Cause of Loss at Cap = 65 (10,000 S1)**:
    - Never retrieved by any blocker (Category A): **843 (37.7%)**
    - Retrieved by blockers, but discarded by cap = 65 (Category B): **1,391 (62.3%)**
    - **Conclusion**: The degradation is predominantly caused by candidate truncation/pruning, NOT blocker retrieval failure!
  - **Per-Blocker Modality Breakdown (10,000 S1, 34,752 matches)**:
    - Exact Name: 28.61% standalone, 0 unique
    - Rare Name Token: 55.91% standalone, 40 unique (0.12%)
    - Char 3-Gram TF-IDF: 88.47% standalone, **1,766 unique (5.08%)**
    - Domain / Brand: 23.98% standalone, 0 unique
    - Address Digits (>=3): 53.71% standalone, **631 unique (1.82%)**
    - Compound 2-Digit Keys: 15.10% standalone, **189 unique (0.54%)**
    - Address Tokens (>=2): 53.32% standalone, **686 unique (1.97%)**
  - **Cap Sweep on 10,000 S1**:
    - Cap 50: 93.23% recall, 42.5 cands/S1, 99.955% red. ratio
    - Cap 65: 93.57% recall, 51.9 cands/S1, 99.945% red. ratio
    - Cap 100: 94.23% recall, 69.5 cands/S1, 99.927% red. ratio
    - Cap 150: 95.03% recall, 87.1 cands/S1, 99.908% red. ratio
    - Cap 250: 96.26% recall, 107.5 cands/S1, 99.887% red. ratio
    - Uncapped: 97.57% recall, 122.2 cands/S1, 99.871% red. ratio
  - **Tiered Candidate Policy (High confidence prioritized)**:
    - Tiered Cap 65: **94.81% recall** (+1.24% recall over flat cap 65, exact same candidate volume of 51.9/S1)
    - Tiered Cap 100: **95.24% recall**
    - Tiered Cap 150: **95.78% recall**

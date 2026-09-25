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

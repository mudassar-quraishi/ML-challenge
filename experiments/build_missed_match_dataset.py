"""Phase 4.1: Build Dedicated Missed-Match Dataset and Failure Categorization.

Extracts all ground-truth matches on 10,000 S1 that were NOT retrieved by any blocker:
- Gathers full record metadata for S1 and Target
- Analyzes scripts, strings, tokens, addresses, and digits
- Categorizes failures into deterministic empirical buckets
- Computes exact frequencies of each failure category
- Saves to experiments/missed_matches_analysis.csv
"""

import os
import sys
import time
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

PROJECT_ROOT = r"c:\Users\mudas\OneDrive\Desktop\ML Challenge"
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CODE_DIR)

from src.config import PipelineConfig
from src.data_loading import load_train_data
from src.normalization import normalize_dataframe, extract_address_digits
from src.blocking import MultiModalCandidateBlocker, extract_core_brand, COMMON_STOP_TOKENS, ADDRESS_STOP_TOKENS

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)


def detect_script(text: str) -> str:
    """Classify text script: latin, devanagari, tamil, bengali, telugu, or mixed."""
    if not text:
        return "empty"
    scripts = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            scripts.add("latin")
        elif "DEVANAGARI" in name:
            scripts.add("devanagari")
        elif "TAMIL" in name:
            scripts.add("tamil")
        elif "BENGALI" in name:
            scripts.add("bengali")
        elif "TELUGU" in name:
            scripts.add("telugu")
        elif "ARABIC" in name:
            scripts.add("arabic")
        elif "CYRILLIC" in name:
            scripts.add("cyrillic")
        else:
            scripts.add("other")
            
    if not scripts:
        return "non_alpha"
    if len(scripts) == 1:
        return list(scripts)[0]
    if "latin" in scripts and len(scripts) == 2:
        other_scripts = scripts - {"latin"}
        return f"mixed_latin_{list(other_scripts)[0]}"
    return "mixed_multilingual"


def categorize_missed_match(
    s1_name: str,
    t_name: str,
    s1_addr: str,
    t_addr: str,
    s1_digits: List[str],
    t_digits: List[str],
    s1_script: str,
    t_script: str
) -> str:
    """Deterministically categorize why a positive pair failed all 6 blockers."""
    # 1. Missing target address evidence
    if not t_addr or len(t_addr.strip()) == 0:
        return "missing_address_evidence"
        
    # 2. Cross-script difference (e.g. Latin vs Indic script)
    is_cross_script = (
        ("latin" in s1_script and any(s in t_script for s in ["devanagari", "tamil", "bengali", "telugu"]))
        or ("latin" in t_script and any(s in s1_script for s in ["devanagari", "tamil", "bengali", "telugu"]))
        or (s1_script != t_script and s1_script != "latin" and t_script != "latin")
    )
    if is_cross_script:
        return "cross_script_name"
        
    # 3. URL / Domain / Social Handle presence
    url_pattern = r"(\.com|\.in|\.org|\.net|\.fr|www\.|@)"
    if re.search(url_pattern, s1_name.lower()) or re.search(url_pattern, t_name.lower()):
        return "url_domain_issue"
        
    # 4. Word-order variation (high token overlap but different sequence)
    s1_tokens = set(s1_name.lower().split()) - COMMON_STOP_TOKENS
    t_tokens = set(t_name.lower().split()) - COMMON_STOP_TOKENS
    if s1_tokens and t_tokens:
        tok_jaccard = len(s1_tokens.intersection(t_tokens)) / len(s1_tokens.union(t_tokens))
        if tok_jaccard >= 0.5:
            return "word_order_variation"
            
    # 5. Diacritics / Accent variation
    s1_nfkd = "".join(c for c in unicodedata.normalize("NFKD", s1_name) if not unicodedata.combining(c)).lower()
    t_nfkd = "".join(c for c in unicodedata.normalize("NFKD", t_name) if not unicodedata.combining(c)).lower()
    if s1_nfkd == t_nfkd and s1_name.lower() != t_name.lower():
        return "diacritics_variation"
        
    # 6. Shared address / DBA Trade-Name mismatch
    # (Completely different business name, but identical or very high address overlap)
    s1_addr_tokens = set(s1_addr.lower().split()) - ADDRESS_STOP_TOKENS
    t_addr_tokens = set(t_addr.lower().split()) - ADDRESS_STOP_TOKENS
    addr_overlap = len(s1_addr_tokens.intersection(t_addr_tokens)) if s1_addr_tokens and t_addr_tokens else 0
    common_digits = set(s1_digits).intersection(set(t_digits))
    name_ratio = fuzz.ratio(s1_name.lower(), t_name.lower())
    
    if name_ratio < 40 and (addr_overlap >= 3 or len(common_digits) >= 1):
        return "dba_trade_name_mismatch"
        
    # 7. Extreme typo / Phonetic spelling difference
    if 30 <= name_ratio < 65:
        return "extreme_typo"
        
    # 8. Numeric variation in address or name
    if common_digits and addr_overlap < 2:
        return "numeric_variation"
        
    # 9. Generic address-only relation
    if addr_overlap >= 2:
        return "address_only_relation"
        
    return "other_mismatch"


def build_missed_match_dataset():
    print("=" * 80)
    print("PHASE 4.1: BUILDING DEDICATED MISSED-MATCH DATASET (10,000 S1 UNIVERSE)")
    print("=" * 80)
    
    cfg = PipelineConfig(
        DATA_ROOT=os.path.join(PROJECT_ROOT, "dataset"),
        LOCAL_SMOKE_TEST=True,
        SMOKE_SAMPLE_SIZE=10000,
        BLOCKING_MODE="multimodal",
        BLOCKING_RETENTION_POLICY="tiered",
        BLOCKING_MAX_CANDIDATES=85
    )
    
    # 1. Load Data
    t0 = time.time()
    df_s1, df_s2, df_s3, gt_train = load_train_data(cfg)
    df_s1 = normalize_dataframe(df_s1)
    df_s2 = normalize_dataframe(df_s2)
    df_s3 = normalize_dataframe(df_s3)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    n_targets = len(df_targets)
    n_s1 = len(df_s1)
    print(f"Data loaded in {time.time() - t0:.1f}s: S1={n_s1}, Targets={n_targets}")
    
    total_true_matches = sum(len(matches) for matches in gt_train.values())
    print(f"Total True Matches: {total_true_matches:,}")
    
    # 2. Build MultiModal Blocker and Generate Candidates
    blocker = MultiModalCandidateBlocker(cfg)
    blocker.build_index(df_targets)
    
    # Generate raw candidates
    print("Generating candidates to find unretrieved true matches...")
    s1_names = df_s1["norm_name"].fillna("").tolist()
    s1_char_matrix = blocker.vectorizer.transform(s1_names)
    chunk_size = 500
    
    target_lookup = {row["entity_id"]: row.to_dict() for _, row in df_targets.iterrows()}
    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in df_s1.iterrows()}
    
    unretrieved_pairs = []
    
    for start_idx in range(0, n_s1, chunk_size):
        end_idx = min(start_idx + chunk_size, n_s1)
        chunk_s1_matrix = s1_char_matrix[start_idx:end_idx]
        sim_matrix = (chunk_s1_matrix * blocker.target_char_matrix.T).tocsr()
        
        for offset, s1_row_idx in enumerate(range(start_idx, end_idx)):
            row = df_s1.iloc[s1_row_idx]
            s1_id = row["entity_id"]
            true_matches = gt_train.get(s1_id, set())
            if not true_matches:
                continue
                
            name = row["norm_name"]
            raw_name = row["business_name"]
            addr = row["norm_address"]
            country = row["norm_country"]
            digits = row.get("address_digits", [])
            
            combined_cands = set()
            
            # Modality A
            if name and name in blocker.exact_name_index:
                combined_cands.update(blocker.exact_name_index[name])
            # Modality B
            tokens = [t for t in set(name.split()) if len(t) >= 3 and t not in COMMON_STOP_TOKENS]
            for t in tokens:
                if t in blocker.token_index:
                    combined_cands.update(blocker.token_index[t])
            # Modality C
            row_vec = sim_matrix[offset]
            if row_vec.nnz > 0:
                above_min = np.where(row_vec.data >= blocker.config.NAME_TFIDF_MIN_SIM)[0]
                for idx in above_min:
                    col_idx = row_vec.indices[idx]
                    combined_cands.add(blocker.target_ids[col_idx])
            # Modality D
            core_brand = extract_core_brand(raw_name)
            if len(core_brand) >= 4 and core_brand in blocker.brand_clean_index:
                combined_cands.update(blocker.brand_clean_index[core_brand])
            # Modality E
            for d in digits[:3]:
                if len(d) >= 3 and (d, country) in blocker.digit_index:
                    combined_cands.update(blocker.digit_index[(d, country)])
            # Compound 2-digit
            addr_tokens = [t for t in set(addr.split()) if len(t) >= 4 and t not in ADDRESS_STOP_TOKENS]
            for d in digits:
                if len(d) == 2:
                    for tok in addr_tokens[:3]:
                        key = (d, tok, country)
                        if key in blocker.compound_digit_index:
                            postings = blocker.compound_digit_index[key]
                            if len(postings) <= 25:
                                combined_cands.update(postings)
            # Modality F
            addr_counts = Counter()
            for tok in addr_tokens:
                if tok in blocker.address_token_index:
                    for tid in blocker.address_token_index[tok]:
                        addr_counts[tid] += 1
            for tid, count in addr_counts.items():
                if count >= 2:
                    combined_cands.add(tid)
                    
            # Check for unretrieved true matches
            for tm in true_matches:
                if tm not in combined_cands:
                    t_record = target_lookup.get(tm, {})
                    unretrieved_pairs.append({
                        "s1_id": s1_id,
                        "target_id": tm,
                        "s1_name": raw_name,
                        "s1_norm_name": name,
                        "target_name": t_record.get("business_name", ""),
                        "target_norm_name": t_record.get("norm_name", ""),
                        "s1_addr": row.get("business_address", ""),
                        "s1_norm_addr": addr,
                        "target_addr": t_record.get("business_address", ""),
                        "target_norm_addr": t_record.get("norm_address", ""),
                        "s1_country": country,
                        "target_country": t_record.get("norm_country", ""),
                        "s1_digits": digits,
                        "target_digits": t_record.get("address_digits", []),
                        "s1_script": detect_script(raw_name),
                        "target_script": detect_script(t_record.get("business_name", ""))
                    })
                    
    print(f"\nExtracted {len(unretrieved_pairs):,} unretrieved true matches ({len(unretrieved_pairs)/total_true_matches*100:.2f}% of all true matches).")
    
    # 3. Categorize Each Unretrieved Match
    categorized_records = []
    category_counter = Counter()
    
    for item in unretrieved_pairs:
        cat = categorize_missed_match(
            s1_name=item["s1_name"],
            t_name=item["target_name"],
            s1_addr=item["s1_norm_addr"],
            t_addr=item["target_norm_addr"],
            s1_digits=item["s1_digits"],
            t_digits=item["target_digits"],
            s1_script=item["s1_script"],
            t_script=item["target_script"]
        )
        item["failure_category"] = cat
        item["name_fuzz_ratio"] = fuzz.ratio(item["s1_norm_name"], item["target_norm_name"])
        categorized_records.append(item)
        category_counter[cat] += 1
        
    df_missed = pd.DataFrame(categorized_records)
    out_csv = os.path.join(PROJECT_ROOT, "experiments", "missed_matches_analysis.csv")
    df_missed.to_csv(out_csv, index=False)
    print(f"Saved missed matches dataset to {out_csv}")
    
    # 4. Display Failure Category Breakdown
    print("\n" + "=" * 80)
    print("MISSED MATCHES FAILURE CATEGORIZATION BREAKDOWN")
    print("=" * 80)
    print(f"{'Failure Category':<30} | {'Count':<8} | {'Percentage':<10} | {'Representative Example':<45}")
    print("-" * 105)
    
    for cat, count in category_counter.most_common():
        pct = count / len(unretrieved_pairs) * 100
        # Find representative example
        sample = next((x for x in categorized_records if x["failure_category"] == cat), None)
        example_str = f"S1: {sample['s1_name'][:20]} | T: {sample['target_name'][:20]}" if sample else ""
        print(f"{cat:<30} | {count:<8} | {pct:>6.2f}%    | {example_str:<45}")
        
    return df_missed, category_counter


if __name__ == "__main__":
    build_missed_match_dataset()

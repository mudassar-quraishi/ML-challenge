"""Pairwise feature engineering for Business Entity Resolution candidate pairs.

Extracts deterministic string, token, digit, country, and interaction features
between Source 1 and candidate Target (Source 2 / 3) entities.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .config import PipelineConfig


FEATURE_NAMES = [
    # Name features
    "exact_name_match",
    "exact_norm_name_match",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_length_diff",
    "name_length_ratio",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    
    # Address features
    "address_exact_match",
    "address_norm_match",
    "address_token_jaccard",
    "address_token_overlap_count",
    "address_digit_jaccard",
    "address_digit_overlap",
    "address_is_empty_target",
    "address_fuzz_ratio",
    
    # Country features
    "country_exact_match",
    
    # Interaction features
    "name_ratio_x_country",
    "name_ratio_x_address_jaccard",
    "name_exact_x_country",
]


def compute_pair_features(s1_meta: dict, target_meta: dict) -> List[float]:
    """Compute rich pairwise features between an S1 entity and a target entity."""
    # 1. Names
    s1_raw_name = s1_meta.get("business_name", "")
    tgt_raw_name = target_meta.get("business_name", "")
    s1_norm_name = s1_meta.get("norm_name", "")
    tgt_norm_name = target_meta.get("norm_name", "")
    
    f_exact_raw = 1.0 if (s1_raw_name and s1_raw_name == tgt_raw_name) else 0.0
    f_exact_norm = 1.0 if (s1_norm_name and s1_norm_name == tgt_norm_name) else 0.0
    
    s1_tokens = set(s1_norm_name.split())
    tgt_tokens = set(tgt_norm_name.split())
    
    intersection = s1_tokens.intersection(tgt_tokens)
    union = s1_tokens.union(tgt_tokens)
    f_tok_jaccard = (len(intersection) / len(union)) if union else 0.0
    f_tok_overlap = float(len(intersection))
    
    l1, l2 = len(s1_norm_name), len(tgt_norm_name)
    f_len_diff = float(abs(l1 - l2))
    f_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0
    
    f_fuzz_ratio = fuzz.ratio(s1_norm_name, tgt_norm_name) / 100.0
    f_fuzz_wratio = fuzz.WRatio(s1_norm_name, tgt_norm_name) / 100.0
    f_tok_sort = fuzz.token_sort_ratio(s1_norm_name, tgt_norm_name) / 100.0
    f_tok_set = fuzz.token_set_ratio(s1_norm_name, tgt_norm_name) / 100.0
    
    # 2. Addresses
    s1_raw_addr = s1_meta.get("business_address", "")
    tgt_raw_addr = target_meta.get("business_address", "")
    s1_norm_addr = s1_meta.get("norm_address", "")
    tgt_norm_addr = target_meta.get("norm_address", "")
    
    f_addr_exact = 1.0 if (s1_raw_addr and s1_raw_addr == tgt_raw_addr) else 0.0
    f_addr_norm = 1.0 if (s1_norm_addr and s1_norm_addr == tgt_norm_addr) else 0.0
    
    tgt_addr_empty = 1.0 if not tgt_norm_addr else 0.0
    
    s1_addr_tokens = set(s1_norm_addr.split())
    tgt_addr_tokens = set(tgt_norm_addr.split())
    addr_inter = s1_addr_tokens.intersection(tgt_addr_tokens)
    addr_union = s1_addr_tokens.union(tgt_addr_tokens)
    f_addr_tok_jaccard = (len(addr_inter) / len(addr_union)) if addr_union else 0.0
    f_addr_tok_overlap = float(len(addr_inter))
    
    s1_digits = set(s1_meta.get("address_digits", []))
    tgt_digits = set(target_meta.get("address_digits", []))
    dig_inter = s1_digits.intersection(tgt_digits)
    dig_union = s1_digits.union(tgt_digits)
    f_dig_jaccard = (len(dig_inter) / len(dig_union)) if dig_union else 0.0
    f_dig_overlap = 1.0 if dig_inter else 0.0
    
    if s1_norm_addr and tgt_norm_addr:
        f_addr_fuzz = fuzz.ratio(s1_norm_addr, tgt_norm_addr) / 100.0
    else:
        f_addr_fuzz = 0.0
        
    # 3. Country (Open world)
    s1_country = s1_meta.get("norm_country", "")
    tgt_country = target_meta.get("norm_country", "")
    f_country_match = 1.0 if (s1_country and s1_country == tgt_country) else 0.0
    
    # 4. Interactions
    f_name_x_country = f_fuzz_wratio * f_country_match
    f_name_x_addr = f_fuzz_wratio * f_addr_tok_jaccard
    f_name_exact_x_country = f_exact_norm * f_country_match
    
    return [
        f_exact_raw,
        f_exact_norm,
        f_tok_jaccard,
        f_tok_overlap,
        f_len_diff,
        f_len_ratio,
        f_fuzz_ratio,
        f_fuzz_wratio,
        f_tok_sort,
        f_tok_set,
        f_addr_exact,
        f_addr_norm,
        f_addr_tok_jaccard,
        f_addr_tok_overlap,
        f_dig_jaccard,
        f_dig_overlap,
        tgt_addr_empty,
        f_addr_fuzz,
        f_country_match,
        f_name_x_country,
        f_name_x_addr,
        f_name_exact_x_country,
    ]


def build_feature_matrix(
    candidate_pairs: List[Tuple[str, str]],
    s1_dict: Dict[str, dict],
    target_dict: Dict[str, dict]
) -> np.ndarray:
    """Build feature matrix for a list of (s1_id, target_id) candidate pairs."""
    features = []
    for s1_id, target_id in candidate_pairs:
        s1_meta = s1_dict.get(s1_id, {})
        tgt_meta = target_dict.get(target_id, {})
        row_feat = compute_pair_features(s1_meta, tgt_meta)
        features.append(row_feat)
        
    if not features:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.array(features, dtype=np.float32)

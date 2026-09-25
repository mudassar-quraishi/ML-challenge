"""Normalization module for Business Entity Resolution.

Deterministic, multilingual, non-destructive normalization for:
- Business names (unicode decomposition, legal suffixes, special symbols)
- Business addresses (standardized street abbreviations, digit extraction)
- Open-world country handling
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple
import pandas as pd


# Common legal suffixes across US, India, France
LEGAL_SUFFIX_MAP = {
    "private limited": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "pvt. ltd": "pvt ltd",
    "private ltd": "pvt ltd",
    "limited": "ltd",
    "ltd.": "ltd",
    "ltd": "ltd",
    "corporation": "corp",
    "corp.": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc.": "inc",
    "inc": "inc",
    "limited liability company": "llc",
    "llc.": "llc",
    "llc": "llc",
    "llp.": "llp",
    "llp": "llp",
    "s.a.r.l.": "sarl",
    "sarl": "sarl",
    "s.a.s.": "sas",
    "sas": "sas",
    "sasu": "sasu",
    "s.c.i.": "sci",
    "sci": "sci",
}

ADDRESS_ABBREV_MAP = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bfl\b": "floor",
    r"\bfl\.\b": "floor",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\br\.\b": "rue",
    r"\br\b": "rue",
    r"\bdoor no\b": "no",
    r"\bh\.no\b": "no",
    r"\bkh no\b": "no",
    r"\bnull\b": "",
}


def strip_accents_and_normalize(text: str) -> str:
    """Normalize unicode and strip combining accents for robust cross-language matching."""
    if not text:
        return ""
    # Normalize unicode to NFKD (separates base characters from accent diacritics)
    nfkd = unicodedata.normalize("NFKD", text)
    # Filter out combining diacritical marks while keeping non-ASCII scripts (Devanagari, Tamil, etc.)
    res = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return res


def normalize_business_name(name: str) -> str:
    """Clean and normalize business name deterministically."""
    if not name or not isinstance(name, str):
        return ""
    
    # 1. Normalize unicode
    text = strip_accents_and_normalize(name).lower()
    
    # 2. Replace ampersand / plus with 'and'
    text = re.sub(r"[\&\+]", " and ", text)
    
    # 3. Remove leading noise punctuation like '-- ', '<< ', etc.
    text = re.sub(r"^[\W_]+", "", text)
    
    # 4. Standardize punctuation to spaces except apostrophes
    text = re.sub(r"[^\w\s\']", " ", text)
    
    # 5. Clean multi-spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # 6. Normalize legal suffixes at end of name
    for suffix, standard in LEGAL_SUFFIX_MAP.items():
        pattern = r"\b" + re.escape(suffix) + r"$"
        if re.search(pattern, text):
            text = re.sub(pattern, standard, text).strip()
            break
            
    return text


def normalize_address(address: str) -> str:
    """Clean and standardize business address."""
    if not address or not isinstance(address, str):
        return ""
    
    # 1. Normalize unicode
    text = strip_accents_and_normalize(address).lower()
    
    # 2. Standardize common abbreviations
    for pattern, replacement in ADDRESS_ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text)
        
    # 3. Clean punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_address_digits(address: str) -> List[str]:
    """Extract sequence of numbers/PINs/postal codes from address."""
    if not address:
        return []
    return re.findall(r"\b\d+\b", address)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns without destroying raw columns."""
    df = df.copy()
    if "business_name" in df.columns:
        df["norm_name"] = df["business_name"].apply(normalize_business_name)
    else:
        df["norm_name"] = ""
        
    if "business_address" in df.columns:
        df["norm_address"] = df["business_address"].apply(normalize_address)
        df["address_digits"] = df["business_address"].apply(extract_address_digits)
    else:
        df["norm_address"] = ""
        df["address_digits"] = [[] for _ in range(len(df))]
        
    if "country" in df.columns:
        df["norm_country"] = df["country"].fillna("").astype(str).str.strip().str.upper()
    else:
        df["norm_country"] = "UNKNOWN"
        
    return df

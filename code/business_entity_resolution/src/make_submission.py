"""Submission generation module for Business Entity Resolution.

Formats and writes the exact official competition TSV outputs:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Guarantees 100% compliance with validate_submission.py:
- Exactly 1 row per test S1 entity
- Empty fields for singletons
- No intra-list duplicate IDs
- Strictly valid prefixes (S2-, S3-)
- Matches guaranteed to be a subset of candidates
"""

import os
from typing import Dict, List, Set, Sequence, Tuple
from .config import PipelineConfig


def format_id_list(ids: Sequence[str]) -> str:
    """Format and deduplicate a sequence of entity IDs into comma-separated string."""
    seen = set()
    cleaned = []
    for item in ids:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)
    return ",".join(cleaned)


def write_submission_files(
    required_s1_ids: Sequence[str],
    candidates_map: Dict[str, List[str]],
    matches_map: Dict[str, List[str]],
    output_dir: str
) -> Tuple[str, str]:
    """Write matching_results.tsv and candidate_pairs.tsv strictly adhering to official format."""
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    print(f"[Submission] Writing official TSV outputs to {output_dir}...")
    
    with open(matching_path, "w", encoding="utf-8") as f_match, \
         open(candidate_path, "w", encoding="utf-8") as f_cand:
             
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in required_s1_ids:
            cands = candidates_map.get(s1_id, [])
            matches = matches_map.get(s1_id, [])
            
            # Ensure matches are strictly a subset of candidates
            valid_cands = [c for c in cands if c.startswith(("S2-", "S3-"))]
            valid_cands_set = set(valid_cands)
            valid_matches = [m for m in matches if m in valid_cands_set]
            
            cand_str = format_id_list(valid_cands)
            match_str = format_id_list(valid_matches)
            
            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            
    print(f"[Submission] Successfully wrote {len(required_s1_ids)} rows:")
    print(f"  - Matching: {matching_path}")
    print(f"  - Candidates: {candidate_path}")
    return matching_path, candidate_path

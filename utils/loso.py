# utils/loso.py
from typing import List, Tuple, Dict, Optional
import os, glob, re
import pandas as pd

# ---- Your collapse rules ----
_COLLAPSE = {
    "CMU":    ("CMU_a_", "CMU_b_"),
    "Leuven": ("Leuven_1_", "Leuven_2_"),
    "MaxMun": ("MaxMun_a_", "MaxMun_b_", "MaxMun_c_", "MaxMun_d_"),
    "UCLA":   ("UCLA_1_", "UCLA_2_"),
    "UM":     ("UM_1_", "UM_2_"),
}

def _collapse_site_from_prefix(prefix: str) -> str:
    for site, prefixes in _COLLAPSE.items():
        if any(prefix.startswith(p) for p in prefixes):
            return site
    # default: 'Caltech_' -> 'Caltech'
    return prefix[:-1] if prefix.endswith('_') else prefix

def _numeric_tail_as_int(s: str) -> Optional[int]:
    m = re.search(r'(\d+)$', str(s))
    return int(m.group(1)) if m else None

def _site_from_file_id(file_id: str) -> str:
    # prefix is everything before the last '_' + underscore
    parts = str(file_id).split('_')
    if len(parts) <= 1:
        return _collapse_site_from_prefix(str(file_id))
    prefix = "_".join(parts[:-1]) + "_"
    return _collapse_site_from_prefix(prefix)

def _build_numid_to_site_from_csv(phenotype_csv: str) -> Dict[int, str]:
    """
    Map int(numeric_id) -> collapsed site using FILE_ID.
    Falls back to site columns only if FILE_ID is unusable.
    """
    mapping: Dict[int, str] = {}
    if not os.path.exists(phenotype_csv):
        return mapping

    df = pd.read_csv(phenotype_csv)

    # Prefer FILE_ID (e.g., 'Pitt_0050003')
    if "FILE_ID" in df.columns:
        for v in df["FILE_ID"].dropna().astype(str).values:
            if v.lower() == "no_filename":
                continue
            num = _numeric_tail_as_int(v)
            if num is None:
                continue
            site = _site_from_file_id(v)
            mapping[num] = site

    # Fallback: try SITE columns + some id column
    if not mapping:
        site_cols = [c for c in df.columns if "SITE" in c.upper()]
        id_cols   = [c for c in df.columns if "SUB" in c.upper() or c.upper().endswith("ID")]
        for _, row in df.iterrows():
            site_val = None
            for sc in site_cols:
                if sc in df.columns and pd.notna(row[sc]):
                    site_val = str(row[sc]).strip()
                    break
            if site_val is None:
                continue
            # collapse directly
            site = _collapse_site_from_prefix(site_val + "_")
            for ic in id_cols:
                if ic in df.columns and pd.notna(row[ic]):
                    digits = re.sub(r"\D+", "", str(row[ic]))
                    if digits:
                        mapping.setdefault(int(digits), site)
    return mapping

def _build_numid_to_site_from_files(abide_root: str, pipeline: str, filt: str, derivative: str) -> Dict[int, str]:
    """
    Fallback: infer site from filenames: e.g., 'Pitt_0050003_rois_cc200.1D'
    Map int(0050003) -> 'Pitt'
    """
    pat = os.path.join(abide_root, pipeline, filt, f"*_{derivative}.1D")
    mapping: Dict[int, str] = {}
    for p in glob.glob(pat):
        base = os.path.basename(p)
        # (prefix)_(digits)_(derivative).1D
        m = re.match(r"(.+?)_(\d+?)_" + re.escape(derivative) + r"\.1D$", base)
        if not m:
            continue
        prefix, num = m.group(1), int(m.group(2))
        site = _collapse_site_from_prefix(prefix + "_")
        mapping[num] = site
    return mapping

def resolve_sites_for_subjects(
    subject_ids: List[str],
    phenotype_csv: str,
    abide_root: str,
    pipeline: str,
    filt: str,
    derivative: str
) -> List[str]:
    """
    Return collapsed site for each subject_id in the SAME ORDER.
    Works when subject_ids are numeric-only (e.g., '50003').
    """
    csv_map  = _build_numid_to_site_from_csv(phenotype_csv)
    file_map = _build_numid_to_site_from_files(abide_root, pipeline, filt, derivative)

    sites: List[str] = []
    for sid in subject_ids:
        # If sid already contains a prefix (e.g., 'Pitt_0050003'), parse directly:
        if "_" in sid:
            parts = sid.split('_')
            prefix = "_".join(parts[:-1]) + "_"
            sites.append(_collapse_site_from_prefix(prefix))
            continue

        # Numeric-only: match by int value (handles zero padding)
        digits = re.sub(r"\D+", "", sid)
        if not digits:
            raise KeyError(f"[LOSO] Cannot parse numeric id from subject_id='{sid}'")
        key = int(digits)

        if key in csv_map:
            sites.append(csv_map[key])
            continue
        if key in file_map:
            sites.append(file_map[key])
            continue

        raise KeyError(
            f"[LOSO] Cannot resolve site for subject_id='{sid}'. "
            f"CSV FILE_ID and derivative filenames yielded no match."
        )
    return sites

def build_loso_splits_from_sites(subject_ids: List[str], sites: List[str]) -> List[Tuple[List[int], List[int], str]]:
    assert len(subject_ids) == len(sites)
    uniq = sorted(set(sites))
    splits: List[Tuple[List[int], List[int], str]] = []
    for s in uniq:
        test_idx  = [i for i, si in enumerate(sites) if si == s]
        train_idx = [i for i, si in enumerate(sites) if si != s]
        if test_idx and train_idx:
            splits.append((train_idx, test_idx, s))
    return splits

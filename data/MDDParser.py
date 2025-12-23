"""data/MDDParser.py

MDD_wwh_667 dataset reader in the SAME style/API as data/ABIDEParser.py,
so you can reuse your existing dataloader.py / train_eval_evgcn.py pipeline
with minimal edits.

Expected extracted folder structure (from data说明.txt):
  <mdd_root>/
    HC/ROISignals_<SUBJECT>.mat
    MDD/ROISignals_<SUBJECT>.mat

Example subject id: S19-2-0016  -> site = "S19"
Labels follow ABIDE convention used by dataloader.py:
  DX_GROUP: 1 = HC, 2 = patient(MDD)
so dataloader.py can keep:
  self.y = y - 1   -> 0/1

How to point to your dataset:
  Option A (recommended): call set_root("/path/to/MDD_wwh_667") once after import.
  Option B: set environment variable MDD_ROOT.

This module implements the functions used by dataloader.py:
  - get_ids()
  - get_subject_score(subject_list, score)
  - get_networks(subject_list, kind, atlas_name=..., variable=...)
  - feature_selection(features, labels, train_ind, fnum)
  - get_static_affinity_adj(features, pd_dict)

Notes:
- If .mat files are MATLAB v7.3 (HDF5), scipy.loadmat will fail.
  We automatically fall back to h5py if available.
- get_networks computes Pearson correlation FC from ROI time series and returns
  the upper-triangle vector per subject (same output shape style as ABIDEParser.get_networks).
"""


from __future__ import annotations

import os
import re
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from scipy.spatial import distance

# Optional deps for .mat reading
try:
    from scipy.io import loadmat  # type: ignore
except Exception:  # pragma: no cover
    loadmat = None

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


# Canonical site order from your data说明.txt (used only for stable sorting).
SITE_ORDER = ["S12", "S15", "S17", "S19", "S21", "S23", "S25"]

# Dataset root (can be overridden by set_root() or env var)
_mdd_root = os.environ.get("MDD_ROOT", "./data/MDD_wwh_667")


def set_root(mdd_root: str) -> None:
    """Set dataset root folder at runtime."""
    global _mdd_root
    _mdd_root = mdd_root


def _parse_subject_id_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = re.match(r"ROISignals_(.+)\.mat$", name)
    if not m:
        raise ValueError(f"Unexpected filename (expected ROISignals_<id>.mat): {name}")
    return m.group(1)


def _site_from_subject_id(subject_id: str) -> str:
    # Example: S19-2-0016 -> S19
    m = re.match(r"(S\d+)", subject_id)
    if m:
        return m.group(1)
    return subject_id.split("-")[0]


def _read_mat_any(path: str) -> Dict[str, np.ndarray]:
    """Read .mat file, supporting classic MAT (scipy.io.loadmat) and v7.3 HDF5 (h5py)."""
    if loadmat is not None:
        try:
            d = loadmat(path)
            out: Dict[str, np.ndarray] = {}
            for k, v in d.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray):
                    out[k] = v
            if out:
                return out
        except Exception:
            pass

    if h5py is None:
        raise RuntimeError(
            f"Failed to read {path} with scipy.io.loadmat, and h5py is not available "
            "(needed for MATLAB v7.3 files)."
        )

    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        def _visit(name, obj):
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                try:
                    out[name] = np.array(obj)
                except Exception:
                    pass
        f.visititems(_visit)

    if not out:
        raise RuntimeError(f"Could not find any dataset arrays in v7.3 mat: {path}")
    return out


def _ensure_time_by_roi(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts).astype(np.float32)
    ts = np.squeeze(ts)
    if ts.ndim != 2:
        raise RuntimeError(f"ROI time series must be 2D after squeeze, got shape={ts.shape}")
    # Heuristic: if rows < cols, transpose to (T, R)
    if ts.shape[0] < ts.shape[1]:
        ts = ts.T
    return ts


def load_roisignals(path: str, key_candidates: Optional[List[str]] = None) -> np.ndarray:
    """Load ROI time series from a .mat file as float32 array of shape (T, R)."""
    if key_candidates is None:
        key_candidates = [
            "ROISignals", "roiSignals", "roisignals", "ROI_Signals",
            "ROISignal", "data", "X"
        ]

    d = _read_mat_any(path)

    for k in key_candidates:
        if k in d:
            arr = d[k]
            try:
                return _ensure_time_by_roi(arr)
            except Exception:
                pass

    # Fallback: pick the first numeric 2D array
    for _, v in d.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            try:
                return _ensure_time_by_roi(v)
            except Exception:
                continue

    raise RuntimeError(f"No suitable ROI time-series found in {path}. Keys={list(d.keys())[:10]}...")


def _fc_upper_vector(ts: np.ndarray, fisher_z: bool = True, eps: float = 1e-6) -> np.ndarray:
    """Pearson corr among ROIs (columns). Return upper-triangle vector."""
    corr = np.corrcoef(ts, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0 + eps, 1.0 - eps)
    if fisher_z:
        corr = 0.5 * np.log((1.0 + corr) / (1.0 - corr))
    iu = np.triu_indices_from(corr, k=1)
    return corr[iu].astype(np.float32)


def _scan_records() -> List[Tuple[str, str, int, str]]:
    """Return list of (subject_id, site, dx_group_label, mat_path). dx_group: 1=HC, 2=MDD."""
    records: List[Tuple[str, str, int, str]] = []
    for folder, dx in [("HC", 1), ("MDD", 2)]:
        patt = os.path.join(_mdd_root, folder, "*.mat")
        for fp in glob.glob(patt):
            sid = _parse_subject_id_from_filename(fp)
            site = _site_from_subject_id(sid)
            records.append((sid, site, dx, fp))

    if not records:
        raise FileNotFoundError(
            f"No .mat files found under {_mdd_root}. Expected HC/*.mat and MDD/*.mat."
        )

    site_rank = {s: i for i, s in enumerate(SITE_ORDER)}
    records.sort(key=lambda r: (site_rank.get(r[1], 10**9), r[2], r[0]))
    return records


def get_ids(num_subjects: Optional[int] = None) -> List[str]:
    """Return list of all subject IDs (stable order)."""
    records = _scan_records()
    ids = [r[0] for r in records]
    if num_subjects is not None:
        ids = ids[:num_subjects]
    return ids


def get_subject_score(subject_list: List[str], score: str) -> Dict[str, str]:
    """Mimic ABIDEParser.get_subject_score: return dict {subject_id: value_as_string}."""
    records = _scan_records()
    id_to_site = {sid: site for sid, site, _, _ in records}
    id_to_dx = {sid: dx for sid, _, dx, _ in records}

    out: Dict[str, str] = {}
    if score == "DX_GROUP":
        for sid in subject_list:
            out[sid] = str(id_to_dx[sid])
        return out
    if score == "SITE_ID":
        for sid in subject_list:
            out[sid] = str(id_to_site[sid])
        return out
    if score == "AGE_AT_SCAN":
        for sid in subject_list:
            out[sid] = "0"
        return out
    if score == "SEX":
        for sid in subject_list:
            out[sid] = "0"
        return out
    raise KeyError(f"Unsupported score={score}. Extend get_subject_score() if needed.")


def get_networks(
    subject_list: List[str],
    kind: str,
    atlas_name: str = "cc200",
    variable: str = "connectivity",
    fisher_z: bool = True,
) -> np.ndarray:
    """Compute FC features from ROI signals for each subject_id in subject_list."""
    records = _scan_records()
    id_to_path = {sid: fp for sid, _, _, fp in records}

    feats: List[np.ndarray] = []
    for sid in subject_list:
        ts = load_roisignals(id_to_path[sid])
        feats.append(_fc_upper_vector(ts, fisher_z=fisher_z))

    return np.stack(feats, axis=0).astype(np.float32)


def feature_selection(features: np.ndarray, labels: np.ndarray, train_ind: List[int], fnum: int) -> np.ndarray:
    """Same as ABIDEParser.feature_selection but safe if fnum > num_features."""
    nfeat = features.shape[1]
    fnum_eff = min(int(fnum), int(nfeat))
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum_eff, step=100, verbose=0)

    tr = np.array(train_ind, dtype=int)
    featureX = features[tr, :]
    featureY = labels[tr]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(features)
    return x_data


def create_affinity_graph_from_scores(scores: List[str], pd_dict: Dict[str, np.ndarray]) -> np.ndarray:
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_vec = pd_dict[l]
        if l in ["AGE_AT_SCAN", "FIQ"]:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_vec[k]) - float(label_vec[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:
                        pass
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_vec[k] == label_vec[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1
    return graph


def get_static_affinity_adj(features: np.ndarray, pd_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Same as ABIDEParser.get_static_affinity_adj."""
    pd_affinity = create_affinity_graph_from_scores(["SEX", "SITE_ID"], pd_dict)

    distv = distance.pdist(features, metric="correlation")
    distm = distance.squareform(distv)
    sigma = np.mean(distm) if np.isfinite(distm).all() else 1.0
    if sigma <= 1e-12:
        sigma = 1.0

    feature_sim = np.exp(-distm ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim
    return adj

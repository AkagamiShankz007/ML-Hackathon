"""Shared data helpers for the final Logistic Regression pipeline (Bayesian optimisation, HMM, export).

Data: the corrected candidate table of src/candidates_v2.py (16 candidate road segments per GPS fix chosen by exact
point-to-segment distance, direction features with a proper angular difference).  Rows are sorted by (trip, observation) so every
observation is a contiguous block of 16 rows.

Label-quality filter: trips whose reference route lies far from the GPS track (median reference distance > 50 m in
artifacts/candidate_summary.csv) have labels that contradict the observations; they are excluded from fitting, cross-validation and
decoder tuning (flag `fit_ok`).  The evaluation partitions are not filtered."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates_v2 import FEATURES                         # the 18 model input columns, in order

K, SEED = 16, 42
CANDIDATES_V2 = ROOT / "artifacts" / "candidates_v2" / "candidate_features_v2.parquet"
LABEL_QUALITY_MAX_MEDIAN_REF_DIST_M = 50.0
IDS = ["trip_id", "observation_id", "candidate_arc", "label", "split"]


def bad_label_trips() -> set:
    s = pd.read_csv(ROOT / "artifacts" / "candidate_summary.csv")
    s["trip_id"] = s.trip_id.astype(str).str.zfill(8)
    return set(s.loc[(s.median_reference_distance_m > LABEL_QUALITY_MAX_MEDIAN_REF_DIST_M) & (s.split == "train"), "trip_id"])


def build_frame() -> pd.DataFrame:
    d = pd.read_parquet(CANDIDATES_V2, columns=IDS + FEATURES)
    d = d.sort_values(["trip_id", "observation_id"], kind="stable").reset_index(drop=True)
    assert (d.groupby(["trip_id", "observation_id"]).size() == K).all()
    d["fit_ok"] = ~d.trip_id.isin(bad_label_trips())
    return d


def fold_assignment(n_splits: int = 5) -> dict:
    """trip_id -> fold (1..n): StratifiedGroupKFold(shuffle, seed 42) grouped by trip on the original (v1) candidate order
    (identical to the fold assignment used for the other model-selection experiments)."""
    raw = pd.read_parquet(ROOT / "artifacts" / "candidate_features.parquet", columns=["trip_id", "split", "label"])
    raw = raw[raw.split == "train"].reset_index(drop=True)
    out = {}
    sp = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)
    for f, (_, vi) in enumerate(sp.split(raw, raw.label.to_numpy(), raw.trip_id.to_numpy(dtype=str)), start=1):
        for t in raw.trip_id.iloc[vi].unique():
            out[t] = f
    return out


def top1_accuracy(label: np.ndarray, score: np.ndarray) -> float:
    Y = np.asarray(label).reshape(-1, K); S = np.asarray(score).reshape(-1, K)
    return float(Y[np.arange(len(Y)), S.argmax(1)].mean())

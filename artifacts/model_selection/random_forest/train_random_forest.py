"""Random Forest candidate scorer (sklearn). p = predict_proba."""
import time, argparse
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
import common

OUT = Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--leaf", type=int, default=10); ap.add_argument("--max_samples", type=float, default=0.3)
    a = ap.parse_args()
    tr, va, te, arr = common.splits()
    cfg = dict(n_estimators=a.n, max_depth=a.depth, min_samples_leaf=a.leaf, max_samples=a.max_samples,
               max_features="sqrt", criterion="gini", random_state=common.SEED)
    rf = RandomForestClassifier(n_jobs=-1, **cfg)
    t0 = time.time(); rf.fit(arr(tr), tr.label.to_numpy()); train_s = time.time() - t0
    pv = rf.predict_proba(arr(va))[:, 1]
    t0 = time.time(); pt = rf.predict_proba(arr(te))[:, 1]; inf_s = time.time() - t0
    common.finish("Random Forest", OUT, cfg, tr, va, te, pv, pt, train_s, inf_s)
    joblib.dump(rf, OUT / "rf.joblib", compress=3)

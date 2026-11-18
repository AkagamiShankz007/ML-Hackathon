"""Shared protocol for classical models: same features, same trip split, same metrics as the MLP."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / "mlp"))
import train_mlp as M
from sklearn.metrics import f1_score

SEED = M.SEED

def splits():
    d = M.load()
    tr_all = d[d.split == "train"]; te = d[d.split == "student_test"]
    trips = np.array(sorted(tr_all.trip_id.unique()))
    val_trips = set(np.random.RandomState(SEED).choice(trips, 12, replace=False))
    va = tr_all[tr_all.trip_id.isin(val_trips)]; tr = tr_all[~tr_all.trip_id.isin(val_trips)]
    mu = tr[M.FEATS].mean(); sd = tr[M.FEATS].std().replace(0, 1)
    arr = lambda df: ((df[M.FEATS] - mu) / sd).to_numpy(np.float32)
    return tr, va, te, arr

def finish(name, out, cfg, tr, va, te, pv, pt, train_s, inf_s):
    ths = np.linspace(0.05, 0.95, 91)
    thr = ths[int(np.argmax([f1_score(va.label, pv >= t, average="macro") for t in ths]))]
    res = dict(model=name, config=cfg, seed=SEED, features=M.FEATS, train_seconds=train_s, inference_seconds=inf_s,
               val=M.metrics(va, pv, pv, thr), student_test=M.metrics(te, pt, pt, thr))
    for k in ("val", "student_test"):
        print(k, {m: round(v, 4) for m, v in res[k].items() if m != "confusion"})
    (out / "results.json").write_text(json.dumps(res, indent=1))
    te.assign(p=pt, rank_logit=pt)[["trip_id", "observation_id", "candidate_arc", "label", "p", "rank_logit"]].to_parquet(out / "student_test_preds.parquet")

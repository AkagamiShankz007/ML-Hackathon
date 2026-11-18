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


def report_student_test(title, te, p, score, thr=0.5):
    """Prints candidate metrics, per-trip Top-1 / route continuity and the confusion matrix on the student-test trips
    (same style as the other test_*.py scripts). te rows: 16 candidates per observation, observation-ordered."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, log_loss,
                                 precision_score, recall_score)
    y = te["label"].to_numpy(); pred = (p >= thr).astype(int)
    d = te[["trip_id", "observation_id", "candidate_arc", "label"]].copy(); d["score"] = score
    top1 = d.loc[d.groupby(["trip_id", "observation_id"])["score"].idxmax()]
    print("-" * 75); print(f"{'Trip ID':<12} | {'GPS Points':<12} | {'Top-1 Accuracy':<18} | Route Continuity"); print("-" * 75)
    conts, tot_ok, tot_pairs = [], 0, 0
    for trip, g in top1.groupby("trip_id"):
        g = g.sort_values("observation_id"); arcs = np.loadtxt(M.DATA / "student_test" / trip / f"{trip}.arcs", dtype=int, ndmin=2)
        a = g["candidate_arc"].to_numpy(); ok = int(((a[:-1] == a[1:]) | (arcs[a[:-1], 1] == arcs[a[1:], 0])).sum())
        conts.append(100.0 * ok / max(len(a) - 1, 1)); tot_ok += ok; tot_pairs += len(a) - 1
        print(f"{trip:<12} | {len(g):<12} | {(g['label'] == 1).mean() * 100:<6.2f}%            | {conts[-1]:<6.2f}%")
    print("-" * 75)
    cm = confusion_matrix(y, pred)
    print("=" * 80); print(f"                    {title}: TEST RESULTS (student_test, 10 trips)"); print("=" * 80)
    print(f"  Overall observation Top-1 accuracy:  {(top1['label'] == 1).mean() * 100:.2f}%")
    print(f"  Overall route continuity (all adjacent pairs): {100.0 * tot_ok / tot_pairs:.2f}%  (mean of trips {np.mean(conts):.2f}%)")
    print(f"  Candidate-level Macro F1:            {f1_score(y, pred, average='macro'):.4f}")
    print(f"  Precision / Recall (traversed class): {precision_score(y, pred, zero_division=0):.4f} / {recall_score(y, pred):.4f}")
    print(f"  Log loss:                            {log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)):.4f}")
    print(f"  Candidate accuracy:                  {accuracy_score(y, pred):.4f}")
    print(f"  Test candidates evaluated:           {len(y):,}"); print("-" * 80)
    print(classification_report(y, pred, target_names=["Non-traversed (0)", "Traversed (1)"]))
    print(f"[Confusion Matrix] TN {cm[0, 0]:,}  FP {cm[0, 1]:,}  FN {cm[1, 0]:,}  TP {cm[1, 1]:,}"); print("=" * 80)

"""MLP candidate scorer for PathSnap (GPU). Usage: python artifacts/model_selection/mlp/train_mlp.py [--epochs N]"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ["distance_m","log_distance","projection","segment_length_m","log_segment_length",
        "bearing_delta","gps_speed_mps","log_gps_gap_s","motion_valid"]
SEED = 42
dev = "cuda"


def load():
    d = pd.read_parquet(ROOT / "artifacts/candidate_features.parquet")
    d = d.sort_values(["trip_id", "observation_id", "distance_m"], kind="stable").reset_index(drop=True)
    d = add_temporal(d)
    d = add_network(d)
    d["obs"] = d.groupby(["trip_id", "observation_id"], sort=False).ngroup()
    g = d.groupby("obs", sort=False)
    # within-observation context features (no label use)
    d["rel_dist"] = d["distance_m"] - g["distance_m"].transform("min")
    d["dist_rank"] = g["distance_m"].rank(method="first") - 1
    d["n_cand"] = g["distance_m"].transform("size")
    d["rank_log"] = np.log1p(d["dist_rank"])
    d["dist_ratio"] = np.log1p(d["distance_m"]) - np.log1p(g["distance_m"].transform("min"))
    d["bearing_best"] = d["bearing_delta"] - g["bearing_delta"].transform("min")
    return d

OFFS = (1, -1, 2, -2, 3, -3, 5, -5)


DATA = ROOT / "Student folder"


def add_network(d):
    """Road-network consistency features: relation of each candidate arc to the nearest-3 candidates of
    neighbouring observations (same arc / directly connected / reverse twin). Uses only arcs topology."""
    out = np.zeros((len(d), 4 * len(NOFF)), np.float32)
    for trip, idx in d.groupby("trip_id", sort=False).indices.items():
        sp = d["split"].iat[idx[0]]
        arcs = np.loadtxt(DATA / sp / trip / f"{trip}.arcs", dtype=np.int64, ndmin=2)
        fr, to = arcs[:, 0], arcs[:, 1]
        A = d["candidate_arc"].to_numpy()[idx].reshape(-1, K)
        F, T = fr[A], to[A]
        n = len(A); cols = []
        for off in NOFF:
            src = np.arange(n) - off  # neighbour observation index
            ok = (src >= 0) & (src < n); src = np.clip(src, 0, n - 1)
            nb = A[src][:, :3]; nF, nT = fr[nb], to[nb]
            for kk in (1, 3):
                nbk, nFk, nTk = nb[:, :kk], nF[:, :kk], nT[:, :kk]
                same = (A[:, :, None] == nbk[:, None, :]).any(2)
                if off > 0:   # neighbour is earlier: it should flow into candidate
                    flow = (nTk[:, None, :] == F[:, :, None]).any(2)
                else:
                    flow = (nFk[:, None, :] == T[:, :, None]).any(2)
                twin = ((nFk[:, None, :] == T[:, :, None]) & (nTk[:, None, :] == F[:, :, None])).any(2)
                for m in (same | flow, twin):
                    cols.append((m & ok[:, None]).astype(np.float32).reshape(-1))
        out[idx] = np.stack(cols, 1)
    for j in range(out.shape[1]): d[f"net{j}"] = out[:, j]
    return d


NOFF = (1, -1, 2, -2)
NETF = [f"net{j}" for j in range(4 * len(NOFF))]


def add_temporal(d):
    key = d[["trip_id", "observation_id", "candidate_arc", "distance_m"]]
    for off in OFFS:
        name = f"p{off}" if off > 0 else f"n{-off}"
        k = key.rename(columns={"distance_m": f"{name}_dist"}).copy()
        k["observation_id"] = k["observation_id"] + off
        d = d.merge(k, on=["trip_id", "observation_id", "candidate_arc"], how="left")
        d[f"{name}_present"] = d[f"{name}_dist"].notna().astype(np.float32)
        d[f"{name}_dist"] = np.log1p(d[f"{name}_dist"].fillna(400.0))
    return d

FEATS = BASE + ["rel_dist","dist_rank","n_cand","rank_log","dist_ratio","bearing_best",] + [f"{n}_{k}" for o in OFFS for n in [f"p{o}" if o > 0 else f"n{-o}"] for k in ("dist", "present")] + NETF


K = 16


class MLP(nn.Module):
    """Candidate MLP with permutation-invariant context pooling over the 16 candidates of an observation.
    Input (B,16,F) -> logits (B,16): head 0 = calibrated candidate probability (BCE), head 1 = ranking (listwise)."""
    def __init__(s, n, h=256, depth=4, p=0.1):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(n, h), nn.SiLU())
        s.blocks = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p)) for _ in range(depth)])
        s.out = nn.Linear(h, 2)
    def forward(s, x):
        z = s.inp(x)
        for b in s.blocks:
            ctx = z.max(1, keepdim=True)[0].expand_as(z)
            z = z + b(torch.cat([z, ctx], -1))
        return s.out(z)


def metrics(df, p, r, thr):
    y = df["label"].to_numpy(); pred = (p >= thr).astype(int)
    Y = y.reshape(-1, K); top = Y[np.arange(len(Y)), r.reshape(-1, K).argmax(1)]
    present = Y.max(1) == 1
    return dict(
        acc=float((pred == y).mean()), precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred)), f1=float(f1_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro")),
        log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        top1_acc=float(top.mean()), top1_acc_given_present=float(top[present].mean()),
        candidate_miss_rate=float(1 - present.mean()),
        confusion=confusion_matrix(y, pred).tolist(), threshold=float(thr))


def predict(model, X, bs=4096):
    """X: (N,16,F) -> (prob, rank_logit) flattened to (N*16,)."""
    model.eval(); P, R = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            o = model(X[i:i + bs]).float(); P.append(torch.sigmoid(o[..., 0])); R.append(o[..., 1])
    return torch.cat(P).cpu().numpy().ravel(), torch.cat(R).cpu().numpy().ravel()


def loss_fn(o, Y, alpha):
    bce = nn.functional.binary_cross_entropy_with_logits(o[..., 0], Y)
    has = Y.sum(1) > 0
    tgt = Y[has] / Y[has].sum(1, keepdim=True)
    lst = -(tgt * torch.log_softmax(o[has][..., 1], -1)).sum(1).mean()
    return bce + alpha * lst


def fit(Xtr, Ytr, Xva, Yva, a):
    model = MLP(Xtr.shape[-1], a.h, a.depth, a.p).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    spe = (len(Xtr) + a.bs - 1) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe)
    yv = Yva.cpu().numpy().ravel(); best, best_state = -1, None; hist = []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Xtr), device=dev); t = 0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(Xtr[idx]).float()
            loss = loss_fn(o, Ytr[idx], a.alpha)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); t += loss.item() * len(idx)
        pv, rv = predict(model, Xva)
        top1 = (yv.reshape(-1, K)[np.arange(len(rv) // K), rv.reshape(-1, K).argmax(1)]).mean()
        print(f"ep {ep+1}/{a.epochs} loss {t/len(Xtr):.5f} val_top1 {top1:.4f} val_logloss {log_loss(yv, np.clip(pv,1e-7,1-1e-7)):.5f}", flush=True)
        hist.append(dict(epoch=ep + 1, train_loss=t / len(Xtr), val_top1=float(top1), val_logloss=float(log_loss(yv, np.clip(pv, 1e-7, 1 - 1e-7)))))
        if top1 > best: best, best_state = top1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.history = hist
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4); ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = load()
    tr_all = d[d.split == "train"]; te = d[d.split == "student_test"]
    trips = np.array(sorted(tr_all.trip_id.unique()))
    val_trips = set(np.random.RandomState(SEED).choice(trips, 12, replace=False))
    va = tr_all[tr_all.trip_id.isin(val_trips)]; tr = tr_all[~tr_all.trip_id.isin(val_trips)]
    mu = tr[FEATS].mean(); sd = tr[FEATS].std().replace(0, 1)
    T = lambda df: torch.tensor(((df[FEATS] - mu) / sd).to_numpy(np.float32), device=dev).view(-1, K, len(FEATS))
    Yt = lambda df: torch.tensor(df.label.to_numpy(np.float32), device=dev).view(-1, K)
    Xtr, Xva, Xte, Ytr, Yva = T(tr), T(va), T(te), Yt(tr), Yt(va)
    print(f"train {len(tr)} val {len(va)} test {len(te)} rows", flush=True)
    t0 = time.time(); model = fit(Xtr, Ytr, Xva, Yva, a); train_s = time.time() - t0
    pv, rv = predict(model, Xva)
    ths = np.linspace(0.05, 0.95, 91)  # macro-F1-optimal threshold chosen on validation trips only
    thr = ths[int(np.argmax([f1_score(va.label, pv >= t, average="macro") for t in ths]))]
    t0 = time.time(); pt, rt = predict(model, Xte); inf_s = time.time() - t0
    res = dict(config=vars(a), seed=SEED, features=FEATS, train_seconds=train_s, inference_seconds=inf_s,
               val=metrics(va, pv, rv, thr), student_test=metrics(te, pt, rt, thr))
    for k in ("val", "student_test"):
        print(k, {m: round(v, 4) for m, v in res[k].items() if m != "confusion"})
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    torch.save(dict(state=model.state_dict(), mu=mu.to_dict(), sd=sd.to_dict(), feats=FEATS,
                    cfg=dict(h=a.h, depth=a.depth, p=a.p), thr=float(thr)), OUT / "mlp.pt")
    te.assign(p=pt, rank_logit=rt)[["trip_id", "observation_id", "candidate_arc", "label", "p", "rank_logit"]].to_parquet(OUT / "student_test_preds.parquet")

if __name__ == "__main__":
    main()

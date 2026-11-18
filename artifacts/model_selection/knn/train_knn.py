"""Distance-weighted KNN candidate scorer, brute-force on GPU. p = weighted fraction of positive neighbours."""
import time, argparse
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
import common

OUT = Path(__file__).resolve().parent

def knn_prob(Xtr, ytr, Xq, k, qb=1024, tb=400_000):
    xn = (Xtr ** 2).sum(1); out = []
    for i in range(0, len(Xq), qb):
        q = Xq[i:i + qb]; qn = (q ** 2).sum(1, keepdim=True)
        bd = bi = None
        for j in range(0, len(Xtr), tb):
            d = qn + xn[j:j + tb][None] - 2 * q @ Xtr[j:j + tb].T
            dv, di = d.topk(k, dim=1, largest=False); di = di + j
            if bd is None: bd, bi = dv, di
            else:
                bd, s = torch.cat([bd, dv], 1).topk(k, dim=1, largest=False)
                bi = torch.cat([bi, di], 1).gather(1, s)
        w = 1.0 / (bd.clamp_min(0).sqrt() + 1e-3)
        out.append(((w * ytr[bi]).sum(1) / w.sum(1)).cpu())
    return torch.cat(out).numpy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=50); a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tr, va, te, arr = common.splits()
    Xtr = torch.tensor(arr(tr), device="cuda"); ytr = torch.tensor(tr.label.to_numpy(np.float32), device="cuda")
    t0 = time.time(); pv = knn_prob(Xtr, ytr, torch.tensor(arr(va), device="cuda"), a.k); val_s = time.time() - t0
    t0 = time.time(); pt = knn_prob(Xtr, ytr, torch.tensor(arr(te), device="cuda"), a.k); inf_s = time.time() - t0
    common.finish("KNN (distance-weighted)", OUT, dict(k=a.k, weights="1/dist", metric="euclidean on standardised features"),
                  tr, va, te, pv, pt, 0.0, inf_s)

"""HMM / Viterbi decoder core (numpy only; shared by the tuning script src/hmm_tuning.py and the inference API).

Hidden states   : at every GPS fix t the state is one of its 16 candidate road segments. State k at fix t is the k-th
                  candidate of that fix, so the candidate <-> road-segment correspondence is preserved.
Emission        : alpha * z, z = prior-corrected logistic logit of the candidate (log-odds that it is the reference arc).
Transition      : log-score looked up from the topological relation between the arc at t and the arc at t+1:
                  same arc | directly connected (head(a) = tail(b)) | connected through 1 / 2 intermediate arcs |
                  U-turn (reverse twin) | not reachable. Scores are free parameters (reference: direct connection = 0).
Gaps            : fix pairs more than GAP_S seconds apart carry no usable motion evidence, so their transition scores are
                  zero and the path falls back to the emissions (a "reset" that is not forced to connect).
Missing/invalid : a path never dies: unreachable pairs get a finite penalty, so if the reference arc is absent from the
                  candidate set the decoder still returns the best available connected candidate."""
from __future__ import annotations
import numpy as np, pandas as pd

K = 16
GAP_S = 5.0
CLASSES = ["none", "same", "hop1", "hop2", "hop3", "uturn"]    # relation classes of (arc at t, arc at t+1)
DEFAULT = dict(alpha=1.0, same=0.0, hop1=0.0, hop2=-1.0, hop3=-2.0, uturn=-3.0, none=-6.0, beta=0.0, gamma=0.0)
DIST_CAP_M = 50.0      # cap of the distance-mismatch term
BACKWARD_M = 2.0       # moving backwards along a one-way arc by more than this is implausible


def edge_codes(arcs: np.ndarray, n_nodes: int):
    """Sorted int64 codes u*n+v of node pairs joined by one arc (D) and by a chain of two arcs (E2)."""
    d = np.unique(arcs[:, 0].astype(np.int64) * n_nodes + arcs[:, 1])
    m = pd.DataFrame(arcs, columns=["u", "v"]).merge(pd.DataFrame(arcs, columns=["v", "w"]), on="v")
    e2 = np.unique(m.u.to_numpy(np.int64) * n_nodes + m.w.to_numpy(np.int64))
    return d, e2


def relation_classes(A: np.ndarray, arcs: np.ndarray, n_nodes: int, d, e2) -> np.ndarray:
    """A: (T,16) candidate arc ids -> (T-1,16,16) uint8 relation class between every candidate at t and t+1."""
    a, b = A[:-1], A[1:]
    tail, head = arcs[:, 0], arcs[:, 1]
    ti, fi = head[a], tail[a]; fj, tj = tail[b], head[b]
    code = ti[:, :, None].astype(np.int64) * n_nodes + fj[:, None, :]
    cls = np.zeros(code.shape, np.uint8)
    cls[np.isin(code, e2)] = 4
    cls[np.isin(code, d)] = 3
    cls[ti[:, :, None] == fj[:, None, :]] = 2
    cls[(fi[:, :, None] == tj[:, None, :]) & (ti[:, :, None] == fj[:, None, :])] = 5     # reverse twin = U-turn
    cls[a[:, :, None] == b[:, None, :]] = 1
    return cls


def transition_table(p: dict) -> np.ndarray:
    return np.array([p["none"], p["same"], p["hop1"], p["hop2"], p["hop3"], p["uturn"]], np.float32)


def viterbi(E: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """E: (T,16) emission log-scores, Tr: (T-1,16,16) transition log-scores [from, to]. Returns best state index per fix."""
    T = len(E); psi = np.zeros((T, K), np.int8); delta = E[0].astype(np.float64); idx = np.arange(K)
    for t in range(1, T):
        c = delta[:, None] + Tr[t - 1]
        psi[t] = c.argmax(0); delta = c[psi[t], idx] + E[t]
    path = np.zeros(T, np.int64); path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def route_delta(A: np.ndarray, P: np.ndarray, L: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Along-road distance (m) travelled between fix t and t+1 for every candidate pair, where it is known:
    same arc: (p_j - p_i) * len ; directly connected arcs: (1 - p_i) * len_i + p_j * len_j ; NaN otherwise.
    P: projection (fraction along the arc, 0..1) and L: arc length in metres, both (T,16)."""
    pi, pj, li, lj = P[:-1], P[1:], L[:-1], L[1:]
    delta = np.full(cls.shape, np.nan, np.float32)
    same = (pj[:, None, :] - pi[:, :, None]) * li[:, :, None]
    hop1 = ((1 - pi) * li)[:, :, None] + (pj * lj)[:, None, :]
    delta[cls == 1] = same[cls == 1]; delta[cls == 2] = hop1[cls == 2]
    return delta


def decode(Z: np.ndarray, cls: np.ndarray, short_gap: np.ndarray, params: dict, aux=None) -> np.ndarray:
    """Z: (T,16) calibrated logits; cls: (T-1,16,16) relation classes; short_gap: (T-1,) bool (gap <= GAP_S).
    aux = (delta, d_gps): along-road distances from route_delta() and the GPS displacement (T-1,) in metres. Where the
    along-road distance is known, the path is penalised by beta * |delta - d_gps| / 10 (capped) and by gamma when it moves
    backwards along the arc (this is what separates an arc from its reverse twin)."""
    if len(Z) == 1:
        return np.array([int(Z[0].argmax())])
    Tr = transition_table(params)[cls]
    if aux is not None and (params.get("beta", 0.0) or params.get("gamma", 0.0)):
        delta, dg = aux; known = np.isfinite(delta)
        mismatch = np.minimum(np.abs(np.where(known, delta, 0.0) - dg[:, None, None]), DIST_CAP_M) / 10.0
        Tr = Tr - params.get("beta", 0.0) * np.where(known, mismatch, 0.0) - params.get("gamma", 0.0) * (known & (delta < -BACKWARD_M))
    return viterbi(params["alpha"] * Z, Tr * short_gap[:, None, None])

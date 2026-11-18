"""Inference core for PathSnap: raw trip files -> candidates -> logistic scores -> (HMM/Viterbi) -> reconstructed route.

Pipeline for one trip (reference route never enters the prediction path; it is only read afterwards for metrics):
  1. clean the raw .track (invalid rows / exact duplicates dropped; decreasing timestamps are rejected)
  2. candidates (16 arcs with the smallest exact point-to-segment distance) + distance/direction features   [src/candidates_v2.py]
  3. logistic-regression pipeline -> calibrated logit z_cal
  4. classifier-only: argmax per fix.  HMM: Viterbi over the 16 candidates per fix        [src/hmm_decoder.py]
  5. connect the predicted arcs into a route (short gaps bridged by shortest network paths)"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import logreg_common as C
import hmm_decoder as H
import candidates_v2 as V2
from build_features import build_reference_labels, clean_track, local_xy

K = C.K
NO_ROAD_M = 150.0          # a fix with no candidate road within this distance is reported as unmatched
BRIDGE_LIMIT_M = 1500.0    # longest shortest-path bridge inserted between two disconnected predicted arcs


@dataclass
class Bundle:
    pipeline: object
    pos_weight: float
    hmm_params: dict
    model_name: str
    emission: str = "logistic-calibrated-logit"


def load_bundle(model_dir: Path) -> Bundle:
    model_dir = Path(model_dir)
    spec = json.loads((model_dir / "feature_spec.json").read_text())
    manifest = json.loads((model_dir / "model_manifest.json").read_text())
    hmm = json.loads((model_dir / "hmm_config.json").read_text())
    assert spec["model_input_order"] == C.FEATURES
    return Bundle(pipeline=joblib.load(model_dir / "logistic_regression_final.joblib"), pos_weight=float(spec["calibration"]["pos_weight"]),
                  hmm_params=hmm["decoder_params"], model_name=manifest["model_name"])


# ---------------------------------------------------------------- raw files
def read_trip(trip_dir: Path, trip_id: str):
    track = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    rp = trip_dir / f"{trip_id}.route"
    route = np.atleast_1d(np.loadtxt(rp, dtype=np.int64, ndmin=1)) if rp.exists() else None
    return track, nodes, arcs, route


def kept_rows(track: np.ndarray) -> np.ndarray:
    """Raw .track row index of every observation kept by clean_track (same masks)."""
    valid = np.isfinite(track[:, :2]).all(1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90) & np.isfinite(track[:, 2])
    dup = np.zeros(len(track), bool)
    if len(track) > 1:
        dup[1:] = np.equal(track[1:], track[:-1]).all(1) & valid[1:] & valid[:-1]
    return np.flatnonzero(valid & ~dup)


# ---------------------------------------------------------------- scoring
def score_trip(bundle: Bundle, track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> dict:
    """Candidate arrays shaped (T,16): arc ids A, calibrated logits Z, projection P, distance D."""
    df = V2.candidates_and_features(track, nodes, arcs); T = len(track)
    z = bundle.pipeline.decision_function(df[C.FEATURES].to_numpy(np.float64)) - math.log(bundle.pos_weight)
    shape = lambda col: df[col].to_numpy().reshape(T, K)
    return dict(A=shape("candidate_arc").astype(np.int64), Z=z.reshape(T, K), P=shape("projection"), D=shape("distance_m"))


def relation(A: np.ndarray, arcs: np.ndarray):
    n_nodes = int(arcs.max()) + 1; d, e2 = H.edge_codes(arcs, n_nodes)
    return H.relation_classes(A, arcs, n_nodes, d, e2) if len(A) > 1 else None


def decode_modes(bundle: Bundle, S: dict, arcs: np.ndarray, track: np.ndarray) -> dict:
    """-> {'classifier': (T,) state index per fix, 'hmm': (T,)}"""
    short = np.diff(track[:, 2]) <= H.GAP_S
    cls = relation(S["A"], arcs)
    return {"classifier": S["Z"].argmax(1), "hmm": H.decode(S["Z"], cls, short, bundle.hmm_params) if cls is not None else S["Z"].argmax(1)}


# ---------------------------------------------------------------- route assembly
class RoadGraph:
    """Directed node graph weighted by arc length (metres) for shortest-path bridging."""
    def __init__(self, nodes, arcs, lon0, lat0):
        x, y = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
        self.arcs, self.n = arcs, len(nodes)
        self.len = np.hypot(x[arcs[:, 1]] - x[arcs[:, 0]], y[arcs[:, 1]] - y[arcs[:, 0]])
        e = pd.DataFrame(dict(u=arcs[:, 0], v=arcs[:, 1], w=self.len, arc=np.arange(len(arcs)))).sort_values("w").drop_duplicates(["u", "v"])
        self.pair_arc = {(int(u), int(v)): int(a) for u, v, a in zip(e.u, e.v, e.arc)}
        self.csr = csr_matrix((np.maximum(e.w.to_numpy(), 1e-3), (e.u.to_numpy(), e.v.to_numpy())), shape=(len(nodes), len(nodes)))

    def bridge(self, a: int, b: int):
        """Arc ids to insert between arc a and arc b (a head -> b tail shortest path) or None if farther than BRIDGE_LIMIT_M."""
        u, v = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        if u == v:
            return []
        dist, pred = dijkstra(self.csr, directed=True, indices=u, return_predecessors=True, limit=BRIDGE_LIMIT_M)
        if not np.isfinite(dist[v]):
            return None
        path = [v]
        while path[-1] != u:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        return [self.pair_arc[(int(p), int(q))] for p, q in zip(path[:-1], path[1:])]


def build_route(pred_arcs: np.ndarray, graph: RoadGraph, nodes: np.ndarray):
    """Collapse repeats, bridge disconnected consecutive arcs. Returns (route arc ids, polylines [[lat,lon],..], n_breaks)."""
    seq = [int(pred_arcs[0])]
    for a in pred_arcs[1:]:
        if int(a) != seq[-1]:
            seq.append(int(a))
    route, groups, cur, breaks = [seq[0]], [], [seq[0]], 0
    for a in seq[1:]:
        prev = route[-1]
        if graph.arcs[prev, 1] == graph.arcs[a, 0]:
            mid = []
        else:
            mid = graph.bridge(prev, a)
        if mid is None:
            breaks += 1; groups.append(cur); cur = [a]; route.append(a)
        else:
            cur += mid + [a]; route += mid + [a]
    groups.append(cur)
    polys = []
    for g in groups:
        pts = [[float(nodes[graph.arcs[g[0], 0], 1]), float(nodes[graph.arcs[g[0], 0], 0])]]
        pts += [[float(nodes[graph.arcs[a, 1], 1]), float(nodes[graph.arcs[a, 1], 0])] for a in g]
        polys.append(pts)
    return route, polys, breaks


# ---------------------------------------------------------------- metrics (server-side, against the internal reference)
def _edit_similarity(a, b) -> float:
    from rapidfuzz.distance import Levenshtein
    return 1.0 - Levenshtein.distance(list(a), list(b)) / max(len(a), len(b), 1)


def evaluate(pred_arcs, A, route_ids, ref_labels, ref_route, arcs, graph, short) -> dict:
    ok = (pred_arcs[:-1] == pred_arcs[1:]) | (arcs[pred_arcs[:-1], 1] == arcs[pred_arcs[1:], 0])
    ref, got = set(ref_route.tolist()), set(route_ids)
    inter = ref & got
    lp = graph.len[list(inter)].sum() if inter else 0.0
    p, r = len(inter) / max(len(got), 1), len(inter) / max(len(ref), 1)
    lprec, lrec = lp / max(graph.len[list(got)].sum(), 1e-9), lp / max(graph.len[list(ref)].sum(), 1e-9)
    f = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
    return dict(top1_accuracy=float((pred_arcs == ref_labels).mean()),
                candidate_miss_rate=float(1.0 - (A == ref_labels[:, None]).any(1).mean()),
                route_continuity=float((ok & short).sum() / max(short.sum(), 1)),
                seq_precision=p, seq_recall=r, seq_f1=f(p, r), length_precision=float(lprec), length_recall=float(lrec),
                edit_similarity=_edit_similarity(route_ids, ref_route.tolist()), n_labelled_observations=int(len(pred_arcs)))


# ---------------------------------------------------------------- full reconstruction
def reconstruct(bundle: Bundle, trip_dir: Path, trip_id: str, mode: str = "hmm", compare: bool = False) -> dict:
    t_all = time.perf_counter(); tm = {}
    t0 = time.perf_counter()
    raw, nodes, arcs, route_ref = read_trip(trip_dir, trip_id)
    track, quality = clean_track(raw, trip_id); rows = kept_rows(raw)
    if np.any((arcs < 0) | (arcs >= len(nodes))):
        raise ValueError("arcs reference unknown nodes")
    tm["prepare_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); S = score_trip(bundle, track, nodes, arcs); tm["score_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); K_sel = decode_modes(bundle, S, arcs, track); tm["decode_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    T = len(track); i = np.arange(T); short = np.diff(track[:, 2]) <= H.GAP_S
    graph = RoadGraph(nodes, arcs, float(track[:, 0].mean()), float(track[:, 1].mean()))
    ref = build_reference_labels(track, nodes, arcs, route_ref) if route_ref is not None else None
    out_modes = {}
    for m, k in K_sel.items():
        pa = S["A"][i, k]; route, polys, breaks = build_route(pa, graph, nodes)
        out_modes[m] = dict(k=k, pa=pa, route=route, polys=polys, breaks=breaks,
                            metrics=evaluate(pa, S["A"], route, ref[0], route_ref, arcs, graph, short) if ref is not None else None)
    sel = out_modes[mode]; k, pa = sel["k"], sel["pa"]
    tail, head = nodes[arcs[pa, 0]], nodes[arcs[pa, 1]]; t = S["P"][i, k][:, None]
    snap = tail + t * (head - tail)                                    # [lon, lat]
    matched = S["D"][i, k] <= NO_ROAD_M
    obs = [dict(index=int(rows[j]), lat=float(track[j, 1]), lon=float(track[j, 0]), t=float(track[j, 2]),
                arc_id=int(pa[j]) if matched[j] else None, snapped=[round(float(snap[j, 1]), 6), round(float(snap[j, 0]), 6)] if matched[j] else None,
                score=round(float(1 / (1 + math.exp(-S["Z"][j, k[j]]))), 5)) for j in range(T)]
    tm["assemble_ms"] = (time.perf_counter() - t0) * 1000; tm["request_total_ms"] = (time.perf_counter() - t_all) * 1000
    return dict(trip_id=trip_id, mode=mode, model_name=bundle.model_name, emission_source=bundle.emission,
                n_observations_raw=int(quality["raw_observations"]), n_observations_used=int(T), n_matched=int(matched.sum()),
                n_segment_breaks=int(sel["breaks"]), route_arc_ids=[int(a) for a in sel["route"]], route_polylines=sel["polys"], observations=obs,
                metrics=sel["metrics"], comparison={m: v["metrics"] for m, v in out_modes.items()} if compare and ref is not None else None,
                timings_ms=tm,
                cleaning_report=dict(n_raw=quality["raw_observations"], n_invalid_dropped=quality["invalid_coordinate_rows"] + quality["invalid_timestamp_rows"],
                                     n_duplicates_dropped=quality["removed_exact_duplicates"], was_time_sorted=True, n_outliers_flagged=0,
                                     n_kept=quality["cleaned_observations"], gaps_over_5s=quality["large_gaps_gt_5s"]))


def network_near_trip(trip_dir: Path, trip_id: str, radius_m: float = 150.0) -> list:
    """Road segments within radius_m of the trajectory as [[lat,lon],[lat,lon]] (for map context)."""
    raw, nodes, arcs, _ = read_trip(trip_dir, trip_id); track, _ = clean_track(raw, trip_id)
    lon0, lat0 = float(track[:, 0].mean()), float(track[:, 1].mean())
    px, py = local_xy(track[:, 0], track[:, 1], lon0, lat0); nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    ax, ay, bx, by = nx[arcs[:, 0]], ny[arcs[:, 0]], nx[arcs[:, 1]], ny[arcs[:, 1]]
    d, _ = cKDTree(np.column_stack([px, py])).query(np.column_stack([(ax + bx) / 2, (ay + by) / 2]))
    keep = d <= radius_m + np.hypot(bx - ax, by - ay) / 2
    return [[[round(float(nodes[a, 1]), 6), round(float(nodes[a, 0]), 6)], [round(float(nodes[b, 1]), 6), round(float(nodes[b, 0]), 6)]]
            for a, b in arcs[keep]]


def trip_summary(trip_dir: Path, trip_id: str) -> dict:
    t = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
    return dict(trip_id=trip_id, n_observations=int(len(t)), duration_s=float(t[:, 2].max() - t[:, 2].min()),
                bbox=[float(t[:, 1].min()), float(t[:, 0].min()), float(t[:, 1].max()), float(t[:, 0].max())])

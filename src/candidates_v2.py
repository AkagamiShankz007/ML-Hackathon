"""Corrected candidate generation + direction features ("v2").  Two problems in the shared v1 pipeline
(src/build_features.py -> point_to_segment_features) are fixed here:

 1. Candidate selection.  v1 takes the 16 arcs whose MIDPOINT is nearest to the GPS fix.  A long arc whose midpoint is far
    away but which passes a few metres from the fix is dropped (4.6 % of student-test fixes and 5-9 % of training fixes lose their
    reference arc this way; the missed arc is a median 6 m from the fix while the 16th candidate is ~80 m away).
    v2 indexes arcs as <=40 m sub-segments, shortlists the 96 nearest sub-segment midpoints and keeps the 16 arcs with the
    smallest EXACT point-to-segment distance.
 2. bearing_delta.  v1 computes np.angle(gps_bearing - arc_bearing) of a REAL number, which can only return 0 or pi, so the
    feature is just the sign of the bearing difference.  v2 uses the wrapped absolute angle |wrap(gps_bearing - arc_bearing)|
    (0 = same direction, pi = opposite) from the one-step displacement and, as a steadier estimate, from a +-2-fix window.

Row layout: observation-major, 16 candidates per fix sorted by distance.  Labels follow the v1 definition
(candidate arc == arc of the nearest reference-route segment).
Usage: python src/candidates_v2.py   -> artifacts/candidates_v2/candidate_features_v2.parquet + candidate_summary_v2.csv"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import LONG_GAP_SECONDS, build_reference_labels, clean_track, load_trip, local_xy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "candidates_v2"
K = 16
SEG_M, K_QUERY = 40.0, 96
MIN_DISP_M, MIN_DISP_SMOOTH_M, WIN = 1.0, 3.0, 2
FEATURES = ["distance_m", "log_distance", "projection", "segment_length_m", "log_segment_length", "bearing_delta", "cos_bearing",
            "gps_speed_mps", "log_gps_gap_s", "motion_valid", "obs_dist_rank", "rel_dist", "speed_aligned",
            "bearing_delta_smooth", "cos_bearing_smooth", "speed_smooth_mps", "smooth_valid", "speed_aligned_smooth"]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def candidates_and_features(track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> pd.DataFrame:
    """track: cleaned (T,3) [lon, lat, t]. Returns a (T*16)-row frame: observation_id, candidate_arc + FEATURES."""
    T = len(track)
    lon0, lat0 = float(track[:, 0].mean()), float(track[:, 1].mean())
    px, py = local_xy(track[:, 0], track[:, 1], lon0, lat0); nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    ax, ay, bx, by = nx[arcs[:, 0]], ny[arcs[:, 0]], nx[arcs[:, 1]], ny[arcs[:, 1]]
    vx, vy = bx - ax, by - ay; vv = vx * vx + vy * vy; length = np.sqrt(vv)
    # ---- candidate selection: nearest sub-segments -> exact point-to-segment distance -> 16 closest arcs
    nsub = np.maximum(1, np.ceil(length / SEG_M)).astype(np.int64)
    arc_of = np.repeat(np.arange(len(arcs)), nsub)
    frac = (np.arange(arc_of.size) - np.repeat(np.cumsum(nsub) - nsub, nsub) + 0.5) / np.repeat(nsub, nsub)
    tree = cKDTree(np.column_stack((ax[arc_of] + frac * vx[arc_of], ay[arc_of] + frac * vy[arc_of])))
    kq = min(K_QUERY, arc_of.size)
    _, idx = tree.query(np.column_stack((px, py)), k=kq)
    idx = np.atleast_2d(idx).reshape(T, kq)
    arcs_q = arc_of[idx]
    order = np.argsort(arcs_q, axis=1, kind="stable"); a_s = np.take_along_axis(arcs_q, order, 1)
    dup = np.zeros_like(a_s, bool); dup[:, 1:] = a_s[:, 1:] == a_s[:, :-1]
    wx, wy = px[:, None] - ax[a_s], py[:, None] - ay[a_s]
    t_s = np.clip((wx * vx[a_s] + wy * vy[a_s]) / np.maximum(vv[a_s], 1e-9), 0.0, 1.0)
    dist_s = np.hypot(wx - t_s * vx[a_s], wy - t_s * vy[a_s]); dist_s[dup] = np.inf
    n_c = min(K, len(arcs))
    sel = np.argsort(dist_s, axis=1, kind="stable")[:, :n_c]
    cand = np.take_along_axis(a_s, sel, 1); dist = np.take_along_axis(dist_s, sel, 1); proj = np.take_along_axis(t_s, sel, 1)
    bad = ~np.isfinite(dist)                                       # fewer than 16 distinct arcs reachable: repeat the closest
    cand[bad] = np.broadcast_to(cand[:, :1], cand.shape)[bad]; dist[bad] = np.broadcast_to(dist[:, :1], dist.shape)[bad]
    proj[bad] = np.broadcast_to(proj[:, :1], proj.shape)[bad]
    if n_c < K:                                                    # tiny network: pad by repeating the closest candidate
        pad = K - n_c
        cand = np.concatenate([cand, np.repeat(cand[:, :1], pad, 1)], 1); dist = np.concatenate([dist, np.repeat(dist[:, :1], pad, 1)], 1)
        proj = np.concatenate([proj, np.repeat(proj[:, :1], pad, 1)], 1)
    # ---- motion (one-step displacement)
    tt = track[:, 2]
    gap = np.diff(tt, prepend=tt[0]); motion_valid = (gap > 0) & (gap <= LONG_GAP_SECONDS)
    dx, dy = np.diff(px, prepend=px[0]), np.diff(py, prepend=py[0]); disp = np.hypot(dx, dy)
    speed = np.zeros(T); speed[motion_valid] = disp[motion_valid] / gap[motion_valid]
    bearing = np.arctan2(dy, dx); dir_valid = motion_valid & (disp >= MIN_DISP_M)
    # ---- smoothed motion over a +-WIN window that contains no long gap
    i = np.arange(T); lo, hi = np.maximum(i - WIN, 0), np.minimum(i + WIN, T - 1)
    long = np.concatenate([[0], (np.diff(tt) > LONG_GAP_SECONDS).astype(int)]); c = np.cumsum(long)
    ddx, ddy, dt = px[hi] - px[lo], py[hi] - py[lo], tt[hi] - tt[lo]
    sdisp = np.hypot(ddx, ddy); smooth_valid = (dt > 0) & (c[hi] - c[lo] == 0) & (sdisp >= MIN_DISP_SMOOTH_M)
    sspeed = np.where(smooth_valid, sdisp / np.maximum(dt, 1e-9), 0.0); sbearing = np.arctan2(ddy, ddx)
    # ---- per-candidate direction agreement with the arc direction (tail -> head)
    arc_b = np.arctan2(vy, vx)[cand]
    d1 = np.where(dir_valid[:, None], np.abs(_wrap(bearing[:, None] - arc_b)), np.pi / 2)
    d2 = np.where(smooth_valid[:, None], np.abs(_wrap(sbearing[:, None] - arc_b)), np.pi / 2)
    cos1, cos2 = np.where(dir_valid[:, None], np.cos(d1), 0.0), np.where(smooth_valid[:, None], np.cos(d2), 0.0)
    rep = lambda v: np.repeat(np.asarray(v, float), K)
    f = lambda m: m.reshape(-1)
    return pd.DataFrame({
        "observation_id": np.repeat(np.arange(T), K), "candidate_arc": f(cand),
        "distance_m": f(dist), "log_distance": np.log1p(f(dist)), "projection": f(proj),
        "segment_length_m": f(length[cand]), "log_segment_length": np.log1p(f(length[cand])),
        "bearing_delta": f(d1), "cos_bearing": f(cos1), "gps_speed_mps": rep(speed), "log_gps_gap_s": rep(np.log1p(gap)),
        "motion_valid": rep(motion_valid), "obs_dist_rank": np.tile(np.arange(1, K + 1), T).astype(float),
        "rel_dist": f(dist - dist[:, :1]), "speed_aligned": f(speed[:, None] * cos1),
        "bearing_delta_smooth": f(d2), "cos_bearing_smooth": f(cos2), "speed_smooth_mps": rep(sspeed),
        "smooth_valid": rep(smooth_valid), "speed_aligned_smooth": f(sspeed[:, None] * cos2)})


def build_trip(split: str, trip_id: str):
    raw, nodes, arcs, route = load_trip(split, trip_id)
    track, _ = clean_track(raw, trip_id)
    frame = candidates_and_features(track, nodes, arcs)
    ref = build_reference_labels(track, nodes, arcs, route) if route is not None else None
    if ref is None:
        return None, None
    ref_arc = ref[0][frame.observation_id.to_numpy()]
    frame["label"] = (frame.candidate_arc.to_numpy() == ref_arc).astype(np.int8)
    frame.insert(0, "trip_id", trip_id); frame.insert(1, "split", split)
    miss = 1.0 - frame.groupby("observation_id").label.max().mean()
    return frame, dict(split=split, trip_id=trip_id, observations=len(track), candidate_miss_rate=float(miss))


def main():
    from joblib import Parallel, delayed
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, d.name) for s in ("train", "student_test") for d in sorted((ROOT / "Student folder" / s).iterdir()) if d.is_dir()]
    t0 = time.time()
    res = Parallel(n_jobs=16)(delayed(build_trip)(s, t) for s, t in jobs)
    frames = [r[0] for r in res if r[0] is not None]; summ = pd.DataFrame([r[1] for r in res if r[1] is not None])
    d = pd.concat(frames, ignore_index=True)
    num = [c for c in FEATURES]; d[num] = d[num].astype(np.float32)
    d.to_parquet(OUT / "candidate_features_v2.parquet"); summ.to_csv(OUT / "candidate_summary_v2.csv", index=False)
    v1 = pd.read_csv(ROOT / "artifacts" / "candidate_summary.csv")[["split", "trip_id", "candidate_miss_rate"]].rename(columns={"candidate_miss_rate": "miss_v1"})
    v1["trip_id"] = v1.trip_id.astype(str).str.zfill(8); summ["trip_id"] = summ.trip_id.astype(str).str.zfill(8)
    m = summ.merge(v1, on=["split", "trip_id"])
    w = m.observations
    print(f"{len(d):,} rows, {d.trip_id.nunique()} trips in {time.time() - t0:.0f}s")
    for s, g in m.groupby("split"):
        print(f"{s}: observation-weighted candidate miss rate v1 {np.average(g.miss_v1, weights=g.observations):.4f} -> v2 {np.average(g.candidate_miss_rate, weights=g.observations):.4f}")


if __name__ == "__main__":
    main()

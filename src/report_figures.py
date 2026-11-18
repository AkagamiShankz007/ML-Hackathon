"""Figures for the pipeline findings in the technical report (all computed from the real data / recorded results).

  candidate_miss_v1_v2.png      candidate miss rate of the v1 (midpoint) vs v2 (exact distance) candidate generation
  bearing_delta_v1_vs_v2.png    distribution of the direction feature before / after the fix
  label_quality_trips.png       median distance between GPS track and reference route per training trip (outlier trips)
  error_breakdown_hmm.png       what the remaining errors of classifier + HMM are (student-test and training out-of-fold)
  top1_by_stage.png             Top-1 accuracy at each stage of the pipeline
Usage: python src/report_figures.py     -> artifacts/report_figures/ + error_breakdown.csv"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts" / "report_figures"; OUT.mkdir(parents=True, exist_ok=True)
import logreg_common as C
import hmm_decoder as H
import hmm_tuning as HV
from build_features import build_reference_labels


def summaries() -> pd.DataFrame:
    v1 = pd.read_csv(ROOT / "artifacts" / "candidate_summary.csv"); v1["trip_id"] = v1.trip_id.astype(str).str.zfill(8)
    v2 = pd.read_csv(ROOT / "artifacts" / "candidates_v2" / "candidate_summary_v2.csv"); v2["trip_id"] = v2.trip_id.astype(str).str.zfill(8)
    return v2.merge(v1[["split", "trip_id", "candidate_miss_rate", "median_reference_distance_m"]].rename(columns={"candidate_miss_rate": "miss_v1"}), on=["split", "trip_id"]) \
             .rename(columns={"candidate_miss_rate": "miss_v2"})


def fig_miss(m):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4)); w = 0.35
    parts = ["train", "student_test"]
    for i, s in enumerate(parts):
        g = m[m.split == s]; v1 = np.average(g.miss_v1, weights=g.observations); v2 = np.average(g.miss_v2, weights=g.observations)
        ax[0].bar(i - w / 2, v1, w, color="#E15759", label="v1: 16 nearest arc midpoints" if i == 0 else None); ax[0].bar(i + w / 2, v2, w, color="#59A14F", label="v2: 16 smallest exact point-to-segment distance" if i == 0 else None)
        ax[0].text(i - w / 2, v1 + .002, f"{v1:.1%}", ha="center"); ax[0].text(i + w / 2, v2 + .002, f"{v2:.1%}", ha="center")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["training (80 trips)", "student-test (10 trips)"]); ax[0].set_ylabel("candidate miss rate (reference arc not among the 16)"); ax[0].legend(); ax[0].set_title("Observation-weighted candidate miss rate")
    for s, c in (("train", "#4C78A8"), ("student_test", "#F28E2B")):
        g = m[m.split == s]; ax[1].scatter(g.miss_v1, g.miss_v2, s=18, c=c, label=s)
    ax[1].plot([0, .7], [0, .7], "k--", lw=.8); ax[1].set_xlabel("miss rate v1"); ax[1].set_ylabel("miss rate v2"); ax[1].set_title("Per trip (points below the line improved)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "candidate_miss_v1_v2.png", dpi=150); plt.close(fig)


def fig_bearing():
    v1 = pd.read_parquet(ROOT / "artifacts" / "candidate_features.parquet", columns=["bearing_delta", "motion_valid", "split"]); v1 = v1[(v1.split == "train") & (v1.motion_valid > 0)].bearing_delta.sample(300_000, random_state=0)
    v2 = pd.read_parquet(C.CANDIDATES_V2, columns=["bearing_delta", "motion_valid", "split"]); v2 = v2[(v2.split == "train") & (v2.motion_valid > 0)].bearing_delta.sample(300_000, random_state=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ax[0].hist(v1, bins=40, color="#E15759"); ax[0].set_title(f"v1 bearing_delta: {v1.nunique()} distinct values (0 or pi)"); ax[0].set_xlabel("radians")
    ax[1].hist(v2, bins=40, color="#59A14F"); ax[1].set_title("v2 bearing_delta: |wrapped angle| of GPS heading vs arc direction"); ax[1].set_xlabel("radians")
    fig.suptitle("Direction feature (training candidates with valid motion)"); fig.tight_layout(); fig.savefig(OUT / "bearing_delta_v1_vs_v2.png", dpi=150); plt.close(fig)


def fig_label_quality(m):
    g = m[m.split == "train"].sort_values("median_reference_distance_m").reset_index(drop=True); bad = g.median_reference_distance_m > C.LABEL_QUALITY_MAX_MEDIAN_REF_DIST_M
    fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(g.index, g.median_reference_distance_m, color=np.where(bad, "#E15759", "#4C78A8")); ax.axhline(C.LABEL_QUALITY_MAX_MEDIAN_REF_DIST_M, ls="--", c="k", lw=.8)
    ax.set_xlabel("training trips (sorted)"); ax.set_ylabel("median distance GPS fix -> nearest reference-route arc (m)"); ax.set_yscale("log")
    for i in g.index[bad]: ax.text(i, g.median_reference_distance_m[i], g.trip_id[i], ha="right", fontsize=8, rotation=90, va="bottom")
    ax.set_title("Label-quality check: trips whose reference route is far from the GPS track are excluded from fitting"); fig.tight_layout(); fig.savefig(OUT / "label_quality_trips.png", dpi=150); plt.close(fig)


def breakdown(trips, params):
    cat = dict(correct=0, reference_arc_not_a_candidate=0, reverse_twin_arc=0, adjacent_arc_on_route=0, other_arc_on_route=0, connected_to_reference=0, off_route=0); n = 0
    for tid, t in trips.items():
        T = len(t["Z"]); k = H.decode(t["Z"], t["cls"], t["short"], params, t["aux"]); pa = t["A"][np.arange(T), k]; arcs = t["arcs"]
        ref, _ = build_reference_labels(t["track"], t["nodes"], arcs, t["route"]); rset = set(t["route"].tolist())
        for i in range(T):
            n += 1; r, p = int(ref[i]), int(pa[i])
            if p == r: cat["correct"] += 1
            elif r not in t["A"][i]: cat["reference_arc_not_a_candidate"] += 1
            elif arcs[p, 0] == arcs[r, 1] and arcs[p, 1] == arcs[r, 0]: cat["reverse_twin_arc"] += 1
            elif p in rset: cat["adjacent_arc_on_route" if (arcs[p, 1] == arcs[r, 0] or arcs[p, 0] == arcs[r, 1]) else "other_arc_on_route"] += 1
            elif arcs[p, 1] == arcs[r, 0] or arcs[p, 0] == arcs[r, 1]: cat["connected_to_reference"] += 1
            else: cat["off_route"] += 1
    return {k: v / n for k, v in cat.items()}, n


def per_trip_breakdown(trips, params) -> pd.DataFrame:
    """Per trip: Top-1 error types of classifier + HMM, and how many reverse-twin errors have the predicted twin arc inside the reference route."""
    rows = []
    for tid, t in trips.items():
        T = len(t["Z"]); k = H.decode(t["Z"], t["cls"], t["short"], params, t["aux"]); pa = t["A"][np.arange(T), k]; arcs = t["arcs"]
        ref, _ = build_reference_labels(t["track"], t["nodes"], arcs, t["route"]); rset = set(t["route"].tolist()); r_ = dict(trip_id=tid, observations=T, correct=0, reverse_twin=0, twin_arc_in_reference_route=0, other_error=0)
        for i in range(T):
            r, p = int(ref[i]), int(pa[i])
            if p == r: r_["correct"] += 1
            elif r in t["A"][i] and arcs[p, 0] == arcs[r, 1] and arcs[p, 1] == arcs[r, 0]: r_["reverse_twin"] += 1; r_["twin_arc_in_reference_route"] += int(p in rset)
            else: r_["other_error"] += 1
        rows.append(r_)
    return pd.DataFrame(rows)


def error_runs(trips, params) -> pd.DataFrame:
    """Per trip: number / length of runs of consecutive wrong fixes, classifier only vs classifier + HMM (explains where decoding does not help)."""
    def runs(wrong):
        d = np.diff(np.concatenate([[0], wrong.astype(int), [0]])); st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return len(st), int((en - st).max()) if len(st) else 0
    rows = []
    for tid, t in trips.items():
        T = len(t["Z"]); arcs = t["arcs"]; ref, _ = build_reference_labels(t["track"], t["nodes"], arcs, t["route"]); rset = set(t["route"].tolist()); r_ = dict(trip_id=tid, observations=T)
        for name, k in (("classifier", t["Z"].argmax(1)), ("hmm", H.decode(t["Z"], t["cls"], t["short"], params, t["aux"]))):
            pa = t["A"][np.arange(T), k]; wrong = pa != ref; n, longest = runs(wrong)
            r_.update({f"{name}_wrong": int(wrong.sum()), f"{name}_error_runs": n, f"{name}_longest_error_run": longest, f"{name}_wrong_arc_on_route_share": float(np.mean([int(a_) in rset for a_ in pa[wrong]])) if wrong.any() else 0.0})
        rows.append(r_)
    return pd.DataFrame(rows)


def fig_errors():
    params = json.loads((ROOT / "artifacts" / "hmm" / "hmm_config.json").read_text())["decoder_params"]
    rows = {}
    for name, split, f, train in (("student-test", "student_test", "test_logreg.parquet", False), ("training (out-of-fold)", "train", "oof_logreg.parquet", True)):
        trips = HV.prepare(HV.attach_geometry(pd.read_parquet(ROOT / "artifacts" / "cache" / f), train), split); rows[name], _ = breakdown(trips, params)
        if not train: per_trip_breakdown(trips, params).to_csv(OUT / "error_breakdown_per_trip_student_test.csv", index=False)
        if not train: error_runs(trips, params).to_csv(OUT / "error_runs_per_trip_student_test.csv", index=False)
    df = pd.DataFrame(rows).T; df.to_csv(OUT / "error_breakdown.csv"); err = df.drop(columns="correct")
    fig, ax = plt.subplots(figsize=(10, 4.8)); left = np.zeros(len(err)); cols = ["#b3b3b3", "#E15759", "#F28E2B", "#EDC948", "#76B7B2", "#9467bd"]
    for c, col in zip(err.columns, cols):
        ax.barh(err.index, err[c], left=left, label=c.replace("_", " "), color=col); left += err[c].to_numpy()
    for i, v in enumerate(left): ax.text(v + .002, i, f"{v:.1%} wrong", va="center", fontsize=9)
    ax.set_xlabel("share of GPS fixes"); ax.set_title("Where the remaining errors of classifier + HMM come from (Top-1 label comparison)"); ax.set_xlim(0, left.max() * 1.18); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3); fig.tight_layout()
    fig.savefig(OUT / "error_breakdown_hmm.png", dpi=150); plt.close(fig)


def fig_stages():
    v1 = pd.read_csv(ROOT / "artifacts" / "v1_original_pipeline" / "hmm" / "comparison.csv"); v2 = pd.read_csv(ROOT / "artifacts" / "hmm" / "comparison.csv")
    g = lambda d, m: float(d[(d.partition == "student_test") & (d.method == m)].top1_accuracy.iloc[0])
    vals = [g(v1, "classifier only"), g(v1, "classifier + HMM (tuned)"), g(v2, "classifier only"), g(v2, "classifier + HMM (tuned)")]
    lab = ["v1 data\nclassifier only", "v1 data\n+ HMM", "corrected data\nclassifier only", "corrected data\n+ HMM (final)"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4)); ax.bar(lab, vals, color=["#9ecae9", "#4C78A8", "#a1d99b", "#59A14F"])
    for i, v in enumerate(vals): ax.text(i, v + .01, f"{v:.1%}", ha="center")
    ax.set_ylim(0, 1); ax.set_ylabel("student-test observation Top-1"); ax.set_title("Logistic regression: Top-1 by pipeline stage"); fig.tight_layout(); fig.savefig(OUT / "top1_by_stage.png", dpi=150); plt.close(fig)


def main():
    m = summaries(); fig_miss(m); fig_bearing(); fig_label_quality(m); fig_errors(); fig_stages(); print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()

"""HMM + Viterbi route reconstruction on top of the final (Bayesian-tuned) Logistic Regression.

Hidden states   : at every GPS fix t the state is one of its 16 candidate road segments (arc ids). State index k at fix t is
                  the k-th candidate of that fix, so the candidate <-> road-segment correspondence is preserved.
Emission        : alpha * z, z = prior-corrected logistic logit of the candidate (log-odds that it is the reference arc).
                  Summed along a path this is the log-likelihood-ratio of "these candidates are the traversed arcs".
Transition      : log-score looked up from the topological relation between the arc at t and the arc at t+1:
                  same arc | directly connected (head(a)=tail(b)) | connected through 1 / 2 intermediate arcs |
                  U-turn (reverse twin) | not reachable.  Every score is a free parameter (reference: direct connection = 0).
Gaps            : fix pairs more than 5 s apart carry no usable motion evidence, so their transition scores are zero
                  (decoder falls back to the emissions, i.e. the path is "reset" without being forced to connect).
Missing/invalid : a path never dies: unreachable pairs get a finite penalty, so if the reference arc is absent from the
                  candidate set the decoder still returns the best available (connected) candidate.
Decoder params  : tuned with Optuna (TPE) on the out-of-fold scores of the 80 training trips; the ten student-test trips
                  are only used for evaluation.
Usage           : python src/hmm_tuning.py [--trials 80]   (needs artifacts/cache/{oof,test}_logreg.parquet from
                  src/bayesian_opt_logistic.py)"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, optuna, folium
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from branca.element import Element

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logreg_common as C
from build_features import clean_track, load_trip

from build_features import local_xy
from hmm_decoder import (CLASSES, DEFAULT, GAP_S, K, decode, edge_codes, relation_classes, route_delta, transition_table, viterbi)  # decoder core

OUT = C.ROOT / "artifacts" / "hmm"; CACHE = C.ROOT / "artifacts" / "cache"


# ---------------------------------------------------------------- data preparation / metrics
def attach_geometry(scores: pd.DataFrame, train: bool) -> pd.DataFrame:
    """Add projection and segment length of every scored candidate (row-aligned with the clean v2 candidate table)."""
    f = C.build_frame(); f = (f[(f.split == "train") & f.fit_ok] if train else f[f.split == "student_test"]).reset_index(drop=True)
    assert len(f) == len(scores) and (f.candidate_arc.to_numpy() == scores.candidate_arc.to_numpy()).all(), "score table misaligned"
    out = scores.copy(); out["projection"] = f.projection.to_numpy(); out["segment_length_m"] = f.segment_length_m.to_numpy()
    return out


def prepare(df: pd.DataFrame, split: str) -> dict:
    trips = {}
    for trip, g in df.groupby("trip_id", sort=True):
        track, nodes, arcs, route = load_trip(split, trip); track, _ = clean_track(track, trip)
        T = len(track)
        assert len(g) == T * K and np.array_equal(g.observation_id.to_numpy()[::K], np.arange(T)), f"misaligned trip {trip}"
        n_nodes = int(arcs.max()) + 1; d, e2 = edge_codes(arcs, n_nodes)
        A = g.candidate_arc.to_numpy(np.int64).reshape(T, K)
        cls = relation_classes(A, arcs, n_nodes, d, e2) if T > 1 else None
        aux = None
        if T > 1 and "projection" in g:
            px, py = local_xy(track[:, 0], track[:, 1], float(track[:, 0].mean()), float(track[:, 1].mean()))
            aux = (route_delta(A, g.projection.to_numpy().reshape(T, K), g.segment_length_m.to_numpy().reshape(T, K), cls),
                   np.hypot(np.diff(px), np.diff(py)).astype(np.float32))
        trips[trip] = dict(A=A, Z=g.z_cal.to_numpy(np.float32).reshape(T, K), Y=g.label.to_numpy().reshape(T, K), arcs=arcs, cls=cls, aux=aux,
                           short=np.diff(track[:, 2]) <= GAP_S, route=route, nodes=nodes, track=track)
    return trips


def continuity(pred_arcs: np.ndarray, arcs: np.ndarray, short: np.ndarray):
    ok = (pred_arcs[:-1] == pred_arcs[1:]) | (arcs[pred_arcs[:-1], 1] == arcs[pred_arcs[1:], 0])
    return int((ok & short).sum()), int(short.sum())


def evaluate(trips: dict, params: dict | None, per_trip: bool = False):
    """params=None -> classifier-only (argmax of the logit). Returns aggregate metrics (and per-trip rows)."""
    hit = obs = c_ok = c_n = 0; rows = []; paths = {}
    tp = fp = fn = 0
    for trip, t in trips.items():
        k = t["Z"].argmax(1) if params is None else decode(t["Z"], t["cls"], t["short"], params, t["aux"])
        T = len(k); pred = t["A"][np.arange(T), k]; paths[trip] = k
        h = int(t["Y"][np.arange(T), k].sum()); hit += h; obs += T
        ok, n = continuity(pred, t["arcs"], t["short"]); c_ok += ok; c_n += n
        ref, got = set(t["route"].tolist()), set(pred.tolist()); i = len(ref & got)
        tp += i; fp += len(got) - i; fn += len(ref) - i
        if per_trip:
            rows.append(dict(trip_id=trip, observations=T, top1_accuracy=h / T, route_continuity=ok / max(n, 1),
                             route_arc_precision=i / max(len(got), 1), route_arc_recall=i / max(len(ref), 1)))
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    m = dict(top1_accuracy=hit / obs, route_continuity=c_ok / c_n, route_arc_precision=p, route_arc_recall=r,
             route_arc_f1=2 * p * r / max(p + r, 1e-9), observations=obs, correct=hit, transitions=c_n)
    return (m, pd.DataFrame(rows), paths) if per_trip else m


# ---------------------------------------------------------------- folium map
def arcs_layer(group, nodes, arcs, ids, color, weight, opacity, name):
    for a in ids:
        e = arcs[int(a)]
        folium.PolyLine([[float(nodes[e[0], 1]), float(nodes[e[0], 0])], [float(nodes[e[1], 1]), float(nodes[e[1], 0])]],
                        color=color, weight=weight, opacity=opacity, tooltip=f"{name} arc {int(a)}").add_to(group)


def make_map(path: Path, trip: str, t: dict, k_clf: np.ndarray, k_hmm: np.ndarray):
    T = len(k_clf); track, nodes, arcs = t["track"], t["nodes"], t["arcs"]
    m = folium.Map(location=[float(track[:, 1].mean()), float(track[:, 0].mean())], zoom_start=14, control_scale=True)
    g_net = folium.FeatureGroup(name="Road network", show=True); arcs_layer(g_net, nodes, arcs, range(len(arcs)), "#777777", 1, .3, "Road network")
    g_ref = folium.FeatureGroup(name="Supplied reference route", show=True); arcs_layer(g_ref, nodes, arcs, t["route"], "#2ca02c", 10, .5, "Reference")
    g_clf = folium.FeatureGroup(name="Logistic regression top-1 (classifier only)", show=False)
    arcs_layer(g_clf, nodes, arcs, t["A"][np.arange(T), k_clf], "#ff7f0e", 3, .9, "Classifier-only")
    g_hmm = folium.FeatureGroup(name="Logistic regression + HMM/Viterbi", show=True)
    arcs_layer(g_hmm, nodes, arcs, t["A"][np.arange(T), k_hmm], "#d62728", 3, 1.0, "HMM")
    g_gps = folium.FeatureGroup(name="GPS trajectory", show=True)
    for seg in np.split(track, np.flatnonzero(np.diff(track[:, 2]) > GAP_S) + 1):
        if len(seg) > 1:
            folium.PolyLine(seg[:, [1, 0]].tolist(), color="#1f77b4", weight=1.5, opacity=.6, tooltip="GPS trajectory (gaps >5 s broken)").add_to(g_gps)
    for lon, lat, _ in track[::max(1, T // 80)]:
        folium.CircleMarker([float(lat), float(lon)], radius=1.8, color="#1f77b4", fill=True, fill_opacity=.7).add_to(g_gps)
    for g in (g_gps, g_net, g_ref, g_clf, g_hmm):
        g.add_to(m)
    m.fit_bounds([[float(track[:, 1].min()), float(track[:, 0].min())], [float(track[:, 1].max()), float(track[:, 0].max())]])
    m.get_root().html.add_child(Element(f"<h4 style='position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px;border:1px solid #777'>"
                                        f"PathSnap route reconstruction | student_test trip {trip}</h4>"))
    m.get_root().html.add_child(Element(
        '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:9px 12px;border:1px solid #777;border-radius:4px;font-size:13px">'
        '<strong>Layers</strong><br><span style="color:#1f77b4">━━</span> GPS fixes<br><span style="color:#777">━━</span> Road network<br>'
        '<span style="color:#2ca02c">━━</span> Reference route<br><span style="color:#ff7f0e">━━</span> Classifier-only top-1<br>'
        '<span style="color:#d62728">━━</span> Logistic + HMM/Viterbi</div>'))
    folium.LayerControl(collapsed=False).add_to(m); m.save(path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--progress-terms", action="store_true", help="also tune the along-road progress/distance-mismatch terms (ablation; no gain measured)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    oof = attach_geometry(pd.read_parquet(CACHE / "oof_logreg.parquet"), True); tst = attach_geometry(pd.read_parquet(CACHE / "test_logreg.parquet"), False)
    t0 = time.time(); tr_trips = prepare(oof, "train"); te_trips = prepare(tst, "student_test")
    print(f"prepared {len(tr_trips)} train / {len(te_trips)} test trips in {time.time() - t0:.0f}s", flush=True)

    def objective(trial):
        p = dict(alpha=trial.suggest_float("alpha", 0.2, 3.0, log=True), same=trial.suggest_float("same", -3, 3),
                 hop1=0.0, hop2=trial.suggest_float("hop2", -8, 3), hop3=trial.suggest_float("hop3", -8, 3),
                 uturn=trial.suggest_float("uturn", -12, 2), none=trial.suggest_float("none", -15, 0))
        if a.progress_terms:
            p.update(beta=trial.suggest_float("beta", 0.0, 2.0), gamma=trial.suggest_float("gamma", 0.0, 15.0))
        m = evaluate(tr_trips, p)
        trial.set_user_attr("route_continuity", m["route_continuity"]); trial.set_user_attr("route_arc_f1", m["route_arc_f1"])
        print(f"trial {trial.number:>2}: top1={m['top1_accuracy']:.4f} cont={m['route_continuity']:.4f} {p}", flush=True)
        return m["top1_accuracy"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=15))
    study.enqueue_trial({k: v for k, v in DEFAULT.items() if k not in ("hop1",) and (a.progress_terms or k not in ("beta", "gamma"))})
    optuna.logging.set_verbosity(optuna.logging.WARNING); study.optimize(objective, n_trials=a.trials)
    best = dict(DEFAULT, **study.best_params); best["hop1"] = 0.0
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs")).to_csv(OUT / "decoder_trials.csv", index=False)

    rows = []
    for split, trips in (("train (out-of-fold)", tr_trips), ("student_test", te_trips)):
        for name, p in (("classifier only", None), ("classifier + HMM (default params)", DEFAULT), ("classifier + HMM (tuned)", best)):
            t0 = time.time(); m = evaluate(trips, p); rows.append(dict(partition=split, method=name, decode_seconds=time.time() - t0, **m))
    cmp = pd.DataFrame(rows); cmp.to_csv(OUT / "comparison.csv", index=False)
    m_clf, pt_clf, paths_clf = evaluate(te_trips, None, per_trip=True)
    m_hmm, pt_hmm, paths_hmm = evaluate(te_trips, best, per_trip=True)
    pt = pt_clf.merge(pt_hmm, on=["trip_id", "observations"], suffixes=("_classifier", "_hmm")); pt.to_csv(OUT / "student_test_per_trip.csv", index=False)
    pred = []
    for trip, t in te_trips.items():
        T = len(t["Z"]); i = np.arange(T)
        pred.append(pd.DataFrame(dict(trip_id=trip, observation_id=i, classifier_arc=t["A"][i, paths_clf[trip]], hmm_arc=t["A"][i, paths_hmm[trip]],
                                      classifier_correct=t["Y"][i, paths_clf[trip]], hmm_correct=t["Y"][i, paths_hmm[trip]])))
    pd.concat(pred).to_csv(OUT / "student_test_hmm_predictions.csv", index=False)
    (OUT / "hmm_config.json").write_text(json.dumps(dict(
        decoder_params=best, default_params=DEFAULT, relation_classes=CLASSES, transition_log_scores=dict(zip(CLASSES, transition_table(best).tolist())),
        gap_threshold_seconds=GAP_S, candidates_per_observation=K, emission="alpha * prior-corrected logistic logit",
        emission_prior_correction="z_cal = decision_function - log(pos_weight)", tuned_on="out-of-fold scores of the 80 training trips (observation Top-1)",
        optimiser="Optuna TPE, seed 42", trials=a.trials, evaluated_on="10 student-test trips (never used for tuning)"), indent=2))
    # ---- plots + map
    fig, ax = plt.subplots(figsize=(8, 4.5)); s = cmp[cmp.partition == "student_test"]; x = np.arange(3); w = .38
    ax.bar(x - w / 2, s.top1_accuracy, w, label="observation Top-1"); ax.bar(x + w / 2, s.route_continuity, w, label="route continuity")
    ax.set_xticks(x); ax.set_xticklabels(["classifier\nonly", "HMM\n(default)", "HMM\n(tuned)"]); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Student-test: classifier vs classifier + HMM")
    for i, (v1, v2) in enumerate(zip(s.top1_accuracy, s.route_continuity)):
        ax.text(i - w / 2, v1 + .01, f"{v1:.3f}", ha="center"); ax.text(i + w / 2, v2 + .01, f"{v2:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(OUT / "classifier_vs_hmm.png", dpi=150); plt.close(fig)
    tr = pd.read_csv(OUT / "decoder_trials.csv"); fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(tr.number, tr.value, "o", alpha=.5)
    ax.plot(tr.number, tr.value.cummax(), "r-"); ax.set_xlabel("trial"); ax.set_ylabel("train OOF Top-1"); ax.set_title("Decoder parameter search"); fig.tight_layout()
    fig.savefig(OUT / "decoder_search_history.png", dpi=150); plt.close(fig)
    trip = min(te_trips, key=lambda x: abs(len(te_trips[x]["Z"]) - np.median([len(v["Z"]) for v in te_trips.values()])))
    make_map(OUT / f"route_reconstruction_trip_{trip}.html", trip, te_trips[trip], paths_clf[trip], paths_hmm[trip])
    print(cmp.to_string()); print("BEST", best)


if __name__ == "__main__":
    main()

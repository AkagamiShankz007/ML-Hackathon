"""Model-selection artifacts for MLP, Random Forest and KNN, in the same layout/protocol as
artifacts/model_selection/bagging: 5-fold StratifiedGroupKFold(shuffle, seed 42) grouped by trip,
candidate Macro-F1 at threshold 0.5, observation Top-1, <=5 s-gap route continuity.
Usage: python src/model_selection_export.py mlp|random_forest|knn
These models use the extended feature set from models/mlp/train_mlp.py (9 base features + context, temporal and
road-network-consistency features), not only the nine base columns."""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, folium, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from branca.element import Element
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "models", "models/mlp", "models/knn"):
    sys.path.insert(0, str(ROOT / p))
import train_mlp as M                      # feature engineering + MLP definition
import train_knn                           # GPU brute-force KNN
import train_bagging as TB                 # teammate's route-continuity helper
from build_features import clean_track, load_trip

SEED, K, FE = 42, M.K, M.FEATS
ID = ["trip_id", "observation_id", "candidate_arc"]
INFO = {
    "mlp": dict(title="MLP", family="Neural network (listwise candidate MLP)", dir="mlp"),
    "random_forest": dict(title="Random Forest", family="Bagged decision-tree ensemble", dir="random_forest"),
    "knn": dict(title="KNN", family="Instance-based (distance-weighted k-nearest neighbours)", dir="knn"),
}
RF_CFG = dict(n_estimators=150, max_depth=24, min_samples_leaf=10, max_samples=0.3, max_features="sqrt",
              criterion="gini", n_jobs=-1)
MLP_CFG = dict(epochs=12, h=256, depth=4, p=0.3, lr=2e-3, bs=512, wd=0.01, alpha=1.0)
KNN_CFG = dict(k=50, weights="1/(euclidean distance+1e-3)", metric="euclidean on standardised features")


def key_config(name):
    if name == "mlp":
        c = MLP_CFG
        return (f"4 residual blocks, width {c['h']}, SiLU+LayerNorm, dropout {c['p']}, max-pool context over the 16 candidates; "
                f"AdamW lr {c['lr']} (OneCycle), wd {c['wd']}, batch {c['bs']}, {c['epochs']} epochs; BCE + listwise softmax loss; "
                f"{len(FE)} features")
    if name == "random_forest":
        return f"150 trees; depth 24; min leaf 10; 30% bootstrap sample per tree; sqrt features per split; Gini; {len(FE)} features"
    return f"k=50; distance-weighted (1/d); Euclidean on standardised features; GPU brute force; {len(FE)} features"


# ---------------------------------------------------------------- model fit/predict
def fit_predict(name, fit_df, ev_list, seed):
    """Returns (list of (prob, score) per frame in ev_list, fit_seconds, model, extra)."""
    mu = fit_df[FE].mean(); sd = fit_df[FE].std().replace(0, 1)
    arr = lambda df: ((df[FE] - mu) / sd).to_numpy(np.float32)
    extra = dict(mu=mu, sd=sd)
    t0 = time.perf_counter()
    if name == "mlp":
        trips = np.array(sorted(fit_df.trip_id.unique()))
        es = set(np.random.RandomState(seed).choice(trips, max(4, len(trips) // 8), replace=False))
        a_df, e_df = fit_df[~fit_df.trip_id.isin(es)], fit_df[fit_df.trip_id.isin(es)]
        T = lambda df: torch.tensor(arr(df), device="cuda").view(-1, K, len(FE))
        Yt = lambda df: torch.tensor(df.label.to_numpy(np.float32), device="cuda").view(-1, K)
        torch.manual_seed(seed)
        model = M.fit(T(a_df), Yt(a_df), T(e_df), Yt(e_df), argparse.Namespace(**MLP_CFG))
        secs = time.perf_counter() - t0
        pred = lambda df: M.predict(model, T(df)); extra["history"] = model.history
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=seed, **RF_CFG).fit(arr(fit_df), fit_df.label.to_numpy())
        secs = time.perf_counter() - t0
        pred = lambda df: (lambda p: (p, p))(model.predict_proba(arr(df))[:, 1])
    else:
        Xtr = torch.tensor(arr(fit_df), device="cuda"); ytr = torch.tensor(fit_df.label.to_numpy(np.float32), device="cuda")
        model, secs = None, time.perf_counter() - t0
        pred = lambda df: (lambda p: (p, p))(train_knn.knn_prob(Xtr, ytr, torch.tensor(arr(df), device="cuda"), KNN_CFG["k"]))
    extra["pred"] = pred
    return [pred(e) for e in ev_list], secs, model, extra


def top1_frame(df, p, score):
    c = df[[*ID, "label"]].copy(); c["positive_probability"] = p; c["ranking_score"] = score
    c = c.sort_values(["trip_id", "observation_id", "candidate_arc"], kind="mergesort")
    sel = c.loc[c.groupby(["trip_id", "observation_id"], sort=False)["ranking_score"].idxmax()].copy()
    sel["correct"] = sel["label"].eq(1)
    return sel.sort_values(["trip_id", "observation_id"], kind="mergesort").reset_index(drop=True), c


# ---------------------------------------------------------------- cross-validation
def cross_validate(name, train, folds_of_trip):
    rows = []
    fold_id = train.trip_id.map(folds_of_trip).to_numpy()
    for f in range(1, 6):
        tr, va = train[fold_id != f], train[fold_id == f]
        outs, secs, model, extra = fit_predict(name, tr, [va], SEED + f)
        pv, sv = outs[0]; y = va.label.to_numpy()
        top = y.reshape(-1, K)[np.arange(len(y) // K), sv.reshape(-1, K).argmax(1)].mean()
        if name == "knn":
            train_f1 = float("nan")   # lazy learner: resubstitution score is trivially inflated (each row is its own neighbour)
        else:
            train_f1 = float(f1_score(tr.label, extra["pred"](tr)[0] >= .5, average="macro", zero_division=0))
        rows.append(dict(fold=f, training_trips=int(tr.trip_id.nunique()), validation_trips=int(va.trip_id.nunique()),
                         training_candidates=len(tr), validation_candidates=len(va), training_macro_f1=train_f1,
                         validation_macro_f1=float(f1_score(y, pv >= .5, average="macro", zero_division=0)),
                         validation_log_loss=float(log_loss(y, np.clip(pv, 1e-7, 1 - 1e-7), labels=[0, 1])),
                         validation_top1_accuracy=float(top), fit_seconds=secs))
        print(f"CV fold {f}/5: Macro F1={rows[-1]['validation_macro_f1']:.4f} Top1={top:.4f} fit={secs:.1f}s", flush=True)
        del model, extra; gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- map / plots
def add_arcs(g, nodes, arcs, ids, color, weight, opacity, label):
    for a in ids:
        if 0 <= int(a) < len(arcs):
            e = arcs[int(a)]
            folium.PolyLine([[float(nodes[e[0], 1]), float(nodes[e[0], 0])], [float(nodes[e[1], 1]), float(nodes[e[1], 0])]],
                            color=color, weight=weight, opacity=opacity, tooltip=f"{label} arc {int(a)}").add_to(g)


def make_map(out, title, trip_id, top1):
    track, nodes, arcs, ref = load_trip("student_test", trip_id); track, _ = clean_track(track, trip_id)
    m = folium.Map(location=[float(track[:, 1].mean()), float(track[:, 0].mean())], zoom_start=14, control_scale=True)
    net = folium.FeatureGroup(name="Road network"); add_arcs(net, nodes, arcs, range(len(arcs)), "#777777", 1, .35, "Road network")
    rg = folium.FeatureGroup(name="Supplied reference route")
    if ref is not None:
        add_arcs(rg, nodes, arcs, ref, "#2ca02c", 3, .85, "Reference")
    pg = folium.FeatureGroup(name=f"{title} top-1 predictions")
    add_arcs(pg, nodes, arcs, top1.loc[top1.trip_id == trip_id].sort_values("observation_id").candidate_arc.to_numpy(),
             "#d62728", 4, .95, f"{title} prediction")
    gg = folium.FeatureGroup(name="GPS trajectory")
    for seg in np.split(track, np.flatnonzero(np.diff(track[:, 2]) > 5.0) + 1):
        if len(seg) > 1:
            folium.PolyLine(seg[:, [1, 0]].tolist(), color="#1f77b4", weight=1.5, opacity=.55,
                            tooltip="GPS trajectory (gaps >5 s broken)").add_to(gg)
    for lon, lat, _ in track[::max(1, len(track) // 60)]:
        folium.CircleMarker([float(lat), float(lon)], radius=1.5, color="#1f77b4", fill=True, fill_opacity=.7).add_to(gg)
    for g in (gg, net, rg, pg):
        g.add_to(m)
    m.fit_bounds([[float(track[:, 1].min()), float(track[:, 0].min())], [float(track[:, 1].max()), float(track[:, 0].max())]])
    m.get_root().html.add_child(Element(
        f"<h4 style='position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px;border:1px solid #777'>"
        f"{title} map-matching | student_test trip {trip_id}</h4>"))
    m.get_root().html.add_child(Element(
        '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:9px 12px;border:1px solid #777;'
        'border-radius:4px;font-size:13px"><strong>PathSnap layers</strong><br><span style="color:#777">━━</span> Road network<br>'
        '<span style="color:#1f77b4">━━</span> GPS trajectory / fixes<br><span style="color:#2ca02c">━━</span> Supplied reference route<br>'
        f'<span style="color:#d62728">━━</span> {title} top-1 prediction</div>'))
    folium.LayerControl(collapsed=False).add_to(m); m.save(out)


def write_plots(out, title, folds, met, conf, extra, name, model):
    fs = folds.validation_macro_f1.to_numpy(); mean, sd = fs.mean(), fs.std(ddof=1)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar([title], [mean], yerr=[sd], capsize=8, color="#4C78A8"); ax.set_ylim(0, 1)
    ax.set_ylabel("Candidate-level Macro F1"); ax.set_title(f"{title}: 5-fold trip-grouped CV Macro F1")
    ax.text(0, min(mean + .03, .97), f"{mean:.3f} ± {sd:.3f}", ha="center"); fig.tight_layout()
    fig.savefig(out / "cv_macro_f1.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5)); vals = [met["student_test_top1_accuracy"], met["student_test_candidate_macro_f1"]]
    ax.bar(["Student-test\nobservation Top-1", "Student-test\ncandidate Macro F1"], vals, color=["#59A14F", "#F28E2B"])
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title(f"{title} student-test metrics (different evaluation units)")
    for i, v in enumerate(vals):
        ax.text(i, v + .02, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(out / "student_test_metrics.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5)); v = met["student_test_candidate_log_loss"]; ax.bar([title], [v], color="#E15759")
    ax.set_ylabel("Candidate-level Log Loss (lower is better)"); ax.set_title(f"{title} student-test Log Loss"); ax.text(0, v, f" {v:.3f}", va="bottom")
    fig.tight_layout(); fig.savefig(out / "student_test_log_loss.png", dpi=170); plt.close(fig)
    d = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=["Non-reference", "Reference"])
    d.plot(cmap="Blues", values_format="d", colorbar=False)
    d.ax_.set_title(f"{title} student-test candidate confusion matrix"); d.figure_.tight_layout()
    d.figure_.savefig(out / "student_test_confusion_matrix.png", dpi=170); plt.close(d.figure_)
    if name == "random_forest":
        imp = pd.DataFrame({"feature": FE, "mean_tree_importance": model.feature_importances_}).sort_values("mean_tree_importance", ascending=False)
        imp.to_csv(out / "feature_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 9)); o = imp.sort_values("mean_tree_importance"); ax.barh(o.feature, o.mean_tree_importance, color="#76B7B2")
        ax.set_xlabel("Mean impurity-based importance"); ax.set_title("Random Forest feature importance (descriptive)"); fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=170); plt.close(fig)
    if name == "mlp":
        h = pd.DataFrame(extra["history"]); h.to_csv(out / "training_history.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(h.epoch, h.train_loss, marker="o"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss (BCE + listwise CE)")
        ax[0].set_title("MLP training loss")
        ax[1].plot(h.epoch, h.val_logloss, marker="o", color="#E15759"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation log loss")
        a2 = ax[1].twinx(); a2.plot(h.epoch, h.val_top1, marker="s", color="#59A14F"); a2.set_ylabel("validation Top-1", color="#59A14F")
        ax[1].set_title("MLP validation (early-stopping trips of the final fit)"); fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=170); plt.close(fig)


TEXT = {
    "mlp": dict(
        why="Each GPS fix has 16 competing candidate segments, and the right choice depends on interactions between distance, direction, segment geometry, speed and the neighbouring fixes. An MLP can learn those nonlinear interactions, and because it is differentiable it can be trained with a ranking (listwise softmax) loss that targets Top-1 directly alongside a calibrated binary head whose probabilities feed the HMM.",
        strengths="**Strengths:** learns nonlinear feature interactions; the context pooling lets every candidate see the other 15 candidates of its fix, which plain per-row classifiers cannot do; two heads give both calibrated probabilities (BCE) and a ranking score; inference is fast on GPU; probabilities plug directly into the HMM emission model.\n\n**Limitations:** needs standardised inputs and a GPU for comfortable training; sensitive to hyper-parameters (dropout, weight decay, epochs) and seed; less interpretable than trees; overfits within a few epochs without regularisation and early stopping; the labels are route-derived, not independently verified; candidate-generation misses cannot be recovered.",
        objective="Loss = binary cross-entropy on the probability head + 1.0 x listwise softmax cross-entropy on the ranking head (target distribution = reference candidate). Optimiser AdamW with a OneCycle schedule, bf16 autocast, early stopping on validation Top-1 using 1/8 of the training trips held out inside each fit."),
    "random_forest": dict(
        why="Candidate scoring depends on nonlinear threshold-like rules over distance, projection, direction and geometry; a forest of decorrelated deep trees captures these without scaling, is robust to outliers and gives vote-fraction probabilities that can rank candidates.",
        strengths="**Strengths:** nonlinear interactions with no scaling; averaging many decorrelated trees stabilises predictions; trains and predicts on CPU; gives feature importances; probabilities are usable by the decoder.\n\n**Limitations:** every candidate row is scored independently, so the forest cannot compare a candidate with the other 15 of the same fix (only the engineered relative features carry that information); vote-fraction probabilities are less calibrated than the MLP's (higher log loss); a larger model file and slower inference than the MLP; route-derived labels; candidate misses cannot be recovered.",
        objective="Each tree greedily minimises Gini impurity; the forest averages the trees' class probabilities. No gradient-based loss or training history."),
    "knn": dict(
        why="KNN is the instance-based baseline: if a candidate's feature vector resembles candidates that were the reference segment in other trips, it is probably correct. It needs no training and shows how far pure similarity gets on this feature space.",
        strengths="**Strengths:** no training phase; no distributional assumptions; simple and interpretable (neighbours can be inspected); probabilities come from the distance-weighted neighbour vote.\n\n**Limitations:** inference is the slowest of all models because every query is compared with all ~3M training rows (brute force on a GPU here; impractical for an API on CPU); Euclidean distance in a standardised space of this many features treats all features alike, including redundant ones, so neighbourhoods are noisy; no way to compare candidates within a fix; probabilities from k=50 votes are coarse and poorly calibrated; needs the whole training set at inference time (no compact model file); route-derived labels; candidate misses cannot be recovered.",
        objective="No loss function is optimised; prediction is the 1/(distance+1e-3)-weighted fraction of positive labels among the 50 nearest training candidates. There is no fitting stage (fit time is only data loading)."),
}


def write_report(out, name, cfg_key, folds, m, map_path):
    t, info = TEXT[name], INFO[name]; title = info["title"]
    def f4(v):
        return "N/A" if v != v else f"{v:.4f}"
    fr = "\n".join(f"| {int(r.fold)} | {int(r.training_trips)} | {int(r.validation_trips)} | {r.validation_macro_f1:.4f} | "
                   f"{f4(r.training_macro_f1)} | {r.validation_top1_accuracy:.4f} | {r.fit_seconds:.1f} |" for r in folds.itertuples())
    gap = m["train_minus_cv_macro_f1"]
    if name == "knn":
        ov = ("KNN has no fitting stage and its resubstitution score is trivially inflated (every training row is its own neighbour), so no in-sample Macro F1 is reported. "
              f"The CV-to-student-test Macro-F1 gap is {m['cv_minus_student_test_macro_f1']:.4f}; the low absolute scores indicate that a plain Euclidean neighbourhood underfits the decision problem rather than overfits it.")
    elif gap >= 0.05:
        ov = f"The in-sample Macro F1 is {gap:.3f} above the grouped CV mean (>=0.05 practical flag), a sign of overfitting; compare with the student-test gap below."
    else:
        ov = f"The in-sample-to-grouped-CV Macro-F1 gap is {gap:.4f}, below the 0.05 practical flag, so there is no strong overfitting signal."
    if name == "mlp":
        ov += " Training loss keeps falling after roughly epoch 8 while validation Top-1 and log loss flatten (see `training_curves.png`), which is why early stopping and dropout 0.3 / weight decay 0.01 are used."
    note = (f" For the MLP, Top-1 with the probability head instead of the ranking head is {m['student_test_top1_accuracy_probability_head']:.4f}."
            if name == "mlp" else "")
    score_word = "ranking-head score" if name == "mlp" else "positive probability"
    rep = f"""# {title} model experiment

## Experiment scope

This is the {title}-only result, produced with the same protocol as the Bagging folder: fit and cross-validated on the 80 training trips with five trip-grouped `StratifiedGroupKFold` folds (shuffle, seed 42); the ten `student_test` trips are used only for development comparison. The evaluator partition is not present and was not used. Thresholded metrics use 0.5.

**Feature-set note:** this model uses the extended {len(FE)}-feature set built by `models/mlp/train_mlp.py` (the nine base columns plus within-fix relative-distance/rank features, distance to the same road segment in the fixes 1, 2, 3 and 5 steps before/after, and road-network consistency flags between neighbouring fixes' top candidates). None of them use labels. Models trained only on the nine base columns are therefore not like-for-like with these results.

## Comparison-table row

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 accuracy | Student-test route continuity | Student-test Macro F1 | Student-test Log Loss | Inference time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| {title} | {info['family']} | {cfg_key} | {m['cv_macro_f1_mean']:.4f} ± {m['cv_macro_f1_std']:.4f} | {m['student_test_top1_accuracy']:.4f} | {m['student_test_route_continuity']:.2%} ({m['route_continuity_connected_transitions']:,}/{m['route_continuity_eligible_transitions']:,}) | {m['student_test_candidate_macro_f1']:.4f} | {m['student_test_candidate_log_loss']:.4f} | {m['student_test_inference_seconds']:.2f} s |

CV and student-test Macro F1 are **candidate-level binary** metrics at threshold 0.5. Top-1 accuracy is **observation-level**: the candidate with the largest {score_word} per GPS fix is compared with the route-derived reference label. Log Loss is candidate-level.{note}

## Cross-validation results

| Fold | Training trips | Validation trips | Validation Macro F1 | Training Macro F1 | Validation Top-1 | Fit time (s) |
|---:|---:|---:|---:|---:|---:|---:|
{fr}

Mean CV Macro F1 is {m['cv_macro_f1_mean']:.4f} (std {m['cv_macro_f1_std']:.4f}); mean CV Top-1 is {m['cv_top1_mean']:.4f} (std {m['cv_top1_std']:.4f}). Top-1 varies between folds because trips differ in candidate miss rate and road-network density. Folds are trip-grouped, so no trip appears on both sides.

## Student-testing results

- Candidate rows: {m['student_test_candidates']:,}; positive rate: {m['student_test_candidate_positive_rate']:.2%}; {m['student_test_trips']} trips, {m['student_test_observations']:,} GPS observations.
- Candidate-level accuracy: {m['student_test_candidate_accuracy']:.4f}; macro precision: {m['student_test_candidate_macro_precision']:.4f}; macro recall: {m['student_test_candidate_macro_recall']:.4f}; macro F1: {m['student_test_candidate_macro_f1']:.4f}.
- Positive-class precision / recall / F1: {m['student_test_positive_precision']:.4f} / {m['student_test_positive_recall']:.4f} / {m['student_test_positive_f1']:.4f}.
- Observation-level Top-1 accuracy: {m['student_test_top1_accuracy']:.4f} ({m['student_test_top1_correct']:,}/{m['student_test_observations']:,}); among observations whose reference arc is in the candidate set: {m['student_test_top1_accuracy_when_candidate_present']:.4f} ({m['student_test_top1_correct']:,}/{m['student_test_candidate_present_observations']:,}).
- Candidate-level Log Loss: {m['student_test_candidate_log_loss']:.4f}.
- Candidate-generation miss rate: {m['student_test_candidate_miss_rate']:.2%} (candidate recall {1 - m['student_test_candidate_miss_rate']:.2%}); these cases cannot be recovered by any classifier.
- Strict directed route continuity: {m['student_test_route_continuity']:.2%} across {m['route_continuity_eligible_transitions']:,} adjacent observation pairs with gaps <=5 s ({m['route_continuity_long_gap_transitions_excluded']:,} longer-gap pairs excluded). Classifier-only, no HMM.
- Final fit time on all training candidates: {m['final_fit_seconds']:.1f} s. Mean inference time over three full student-test passes: {m['student_test_inference_seconds']:.2f} s for {m['student_test_candidates']:,} candidate rows ({m['student_test_inference_ms_per_candidate']:.4f} ms/candidate).

Plots use the actual held-out predictions. `{map_path.name}` overlays GPS fixes, the road network, the reference route and the {title} top-1 predictions.

## Why consider {title} for PathSnap?

{t['why']}

## Strengths and limitations for this task

{t['strengths']}

## Configuration and training objective

Key configuration: {cfg_key}. Full settings are in `config.json`. Random seed {SEED}.

{t['objective']}

## Overfitting assessment

{ov}

Mean CV Macro F1 {m['cv_macro_f1_mean']:.4f} vs student-test Macro F1 {m['student_test_candidate_macro_f1']:.4f} (gap {m['cv_minus_student_test_macro_f1']:.4f}). Student-test is for development comparison, not evaluator selection.

## Reproduce

`python src/model_selection_export.py {name}` (requires `artifacts/candidate_features.parquet` and `Student folder/`; the MLP and KNN stages use a CUDA GPU).
"""
    (out / f"{name}_results.md").write_text(rep, encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    name = sys.argv[1]; info = INFO[name]; title = info["title"]
    out = ROOT / "artifacts" / "model_selection" / info["dir"]; out.mkdir(parents=True, exist_ok=True)
    # teammates' fold assignment (original candidate order, StratifiedGroupKFold shuffle seed 42)
    raw = pd.read_parquet(ROOT / "artifacts" / "candidate_features.parquet", columns=["trip_id", "split", "label"])
    raw = raw[raw.split == "train"].reset_index(drop=True)
    folds_of_trip = {}
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for f, (_, vi) in enumerate(splitter.split(raw, raw.label.to_numpy(), raw.trip_id.to_numpy(dtype=str)), start=1):
        for t in raw.trip_id.iloc[vi].unique():
            folds_of_trip[t] = f
    del raw
    d = M.load(); train = d[d.split == "train"].reset_index(drop=True); te = d[d.split == "student_test"].reset_index(drop=True)
    print(f"{title}: {len(train):,} train / {len(te):,} test candidates, {len(FE)} features", flush=True)
    folds = cross_validate(name, train, folds_of_trip); folds.to_csv(out / "cv_fold_metrics.csv", index=False)

    _, fit_s, model, extra = fit_predict(name, train, [], SEED)
    inf = []
    for _ in range(3):
        t0 = time.perf_counter(); pt, st = extra["pred"](te)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf.append(time.perf_counter() - t0)
    inf_s = float(np.mean(inf)); y = te.label.to_numpy(); pred = (pt >= .5).astype(np.int8)
    top1, cands = top1_frame(te, pt, st)
    route = TB.measure_route_continuity(top1)
    train_f1 = float("nan") if name == "knn" else float(f1_score(train.label, extra["pred"](train)[0] >= .5, average="macro", zero_division=0))
    has_ref = te.groupby(["trip_id", "observation_id"]).label.max(); n_obs = len(top1); n_present = int(has_ref.sum())
    top1_corr = int(top1.correct.sum()); conf = confusion_matrix(y, pred, labels=[0, 1])
    mp, mr, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], average="macro", zero_division=0)
    pp, pr, pf, _ = precision_recall_fscore_support(y, pred, labels=[1], average=None, zero_division=0)
    Yr = y.reshape(-1, K); prob_top = Yr[np.arange(len(Yr)), pt.reshape(-1, K).argmax(1)].mean()
    obs_counts = te.groupby("trip_id").observation_id.nunique(); med = float(obs_counts.median())
    map_trip = min(obs_counts.index, key=lambda t: (abs(float(obs_counts[t]) - med), t))
    cvm = folds.validation_macro_f1; test_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    base_cfg = {"MLP": MLP_CFG, "Random Forest": RF_CFG, "KNN": KNN_CFG}[title]
    cfg = dict(base_cfg, features=FE, n_features=len(FE), random_state=SEED, key_configuration=key_config(name),
               cv="StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), grouped by trip_id",
               primary_cv_metric="candidate-level macro F1 at threshold 0.5",
               preprocessing="z-score standardisation fit on the fitting trips only")
    m = dict(model=title, algorithm_family=info["family"], random_state=SEED, cv_splits=5, cv_grouping="trip_id",
             cv_macro_f1_mean=float(cvm.mean()), cv_macro_f1_std=float(cvm.std(ddof=1)),
             cv_top1_mean=float(folds.validation_top1_accuracy.mean()), cv_top1_std=float(folds.validation_top1_accuracy.std(ddof=1)),
             cv_log_loss_mean=float(folds.validation_log_loss.mean()),
             cv_training_macro_f1_mean=float(folds.training_macro_f1.mean()), training_candidate_macro_f1=train_f1,
             training_candidate_positive_rate=float(train.label.mean()),
             student_test_trips=int(te.trip_id.nunique()), student_test_candidates=len(te), student_test_observations=n_obs,
             student_test_candidate_positive_rate=float(y.mean()), student_test_candidate_accuracy=float(accuracy_score(y, pred)),
             student_test_candidate_macro_precision=float(mp), student_test_candidate_macro_recall=float(mr),
             student_test_candidate_macro_f1=test_f1,
             student_test_positive_precision=float(pp[0]), student_test_positive_recall=float(pr[0]), student_test_positive_f1=float(pf[0]),
             student_test_candidate_log_loss=float(log_loss(y, np.clip(pt, 1e-7, 1 - 1e-7), labels=[0, 1])),
             student_test_candidate_miss_rate=float(1 - has_ref.mean()), student_test_candidate_present_observations=n_present,
             student_test_top1_correct=top1_corr, student_test_top1_accuracy=float(top1.correct.mean()),
             student_test_top1_accuracy_probability_head=float(prob_top),
             student_test_top1_accuracy_when_candidate_present=float(top1_corr / n_present),
             student_test_route_continuity=route["continuity"], route_continuity_connected_transitions=route["connected_transitions"],
             route_continuity_eligible_transitions=route["eligible_transitions"],
             route_continuity_long_gap_transitions_excluded=route["long_gap_transitions_excluded"],
             route_continuity_definition=route["definition"],
             final_fit_seconds=float(fit_s), student_test_inference_seconds=inf_s,
             student_test_inference_ms_per_candidate=float(inf_s * 1000 / len(te)), inference_repeat_seconds=[float(v) for v in inf],
             cv_total_fit_seconds=float(folds.fit_seconds.sum()),
             train_minus_cv_macro_f1=None if train_f1 != train_f1 else float(train_f1 - cvm.mean()),
             cv_minus_student_test_macro_f1=float(cvm.mean() - test_f1),
             confusion_matrix_labels=["non_reference", "reference"], confusion_matrix=conf.tolist(), map_trip_id=str(map_trip),
             candidate_threshold=0.5, student_test_top1_score="ranking head" if name == "mlp" else "positive probability",
             configuration=cfg)
    top1.to_csv(out / "student_test_top1_predictions.csv", index=False)
    cands.to_csv(out / "student_test_candidate_predictions.csv", index=False)
    if name == "mlp":
        torch.save(dict(state=model.state_dict(), mu=extra["mu"].to_dict(), sd=extra["sd"].to_dict(), feats=FE,
                        cfg=dict(h=MLP_CFG["h"], depth=MLP_CFG["depth"], p=MLP_CFG["p"])), out / "mlp_model.pt")
        m["model_file"] = (out / "mlp_model.pt").relative_to(ROOT).as_posix()
    elif name == "random_forest":
        joblib.dump(model, out / "random_forest_model.joblib", compress=3)
        m["model_file"] = (out / "random_forest_model.joblib").relative_to(ROOT).as_posix()
    else:
        m["model_file"] = "none (lazy learner; inference needs the training candidates)"
    html = out / f"{name}_student_test_trip_{map_trip}.html"; make_map(html, title, str(map_trip), top1)
    m["prediction_map"] = html.relative_to(ROOT).as_posix()
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    row = dict(model=title, algorithm_family=info["family"], key_configuration=key_config(name), cv_macro_f1=m["cv_macro_f1_mean"],
               cv_macro_f1_std=m["cv_macro_f1_std"], student_test_top1_accuracy=m["student_test_top1_accuracy"],
               student_test_route_continuity=route["continuity"], student_test_candidate_macro_f1=test_f1,
               student_test_candidate_log_loss=m["student_test_candidate_log_loss"], student_test_inference_seconds=inf_s,
               final_fit_seconds=float(fit_s))
    pd.DataFrame([row]).to_csv(out / "comparison_table_row.csv", index=False)
    write_plots(out, title, folds, m, conf, extra, name, model)
    write_report(out, name, key_config(name), folds, m, html)
    print(json.dumps({k: v for k, v in m.items() if k != "configuration"}, indent=1, default=str))


if __name__ == "__main__":
    main()

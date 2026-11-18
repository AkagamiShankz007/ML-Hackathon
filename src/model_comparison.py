"""Unified model comparison: every model, same data, same folds, same metrics.

Data       : corrected v2 candidates (src/candidates_v2.py), 18 features, clean training trips (label-quality filter in
             src/logreg_common.py), 16 candidates per GPS fix.
CV         : StratifiedGroupKFold(5, shuffle, seed 42) grouped by trip on the training partition (fold_assignment()).
Student set: the ten student_test trips, used only for development-stage comparison.
Metrics    : candidate level (threshold 0.5): accuracy, precision, recall, F1, Macro-F1, log loss, confusion matrix;
             observation level: Top-1 accuracy (argmax of the positive probability per GPS fix), candidate miss rate;
             route continuity (adjacent fixes <= 5 s apart whose predicted arcs are identical or directly connected);
             fit and inference time.  The in-sample Macro-F1 of the final fit (random sample of 20,000 training fixes from all trips) is recorded as overfitting evidence.
Outputs    : artifacts/model_comparison/<model>/{metrics.json,config.json,cv_fold_metrics.csv,confusion_matrix.png,
             student_test_top1_predictions.csv[,training_history.csv,feature_importance.csv]}
             artifacts/cache/{oof,test}_<model>.parquet  (prior-corrected logits for the HMM stage)
Usage      : python src/model_comparison.py [model ...]      models: logistic_regression gaussian_nb decision_tree
             random_forest lightgbm bagging mlp knn   (default: all)"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "artifacts/model_selection/mlp", "artifacts/model_selection/knn"):
    sys.path.insert(0, str(ROOT / p))
import logreg_common as C
from build_features import clean_track, load_trip

K, FE, SEED = C.K, C.FEATURES, C.SEED
OUT = ROOT / "artifacts" / "model_comparison"; CACHE = ROOT / "artifacts" / "cache"
MODELS = ["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "lightgbm", "bagging", "mlp", "knn"]
TITLE = {"logistic_regression": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest", "lightgbm": "LightGBM", "bagging": "Bagging (trees)", "mlp": "MLP", "knn": "KNN"}
FAMILY = {"logistic_regression": "Linear / probabilistic", "gaussian_nb": "Probabilistic (generative)", "decision_tree": "Tree-based",
          "random_forest": "Ensemble (bagged trees, random subspaces)", "lightgbm": "Boosting", "bagging": "Ensemble (bootstrap aggregation)",
          "mlp": "Neural network", "knn": "Instance-based"}
CONFIG = {
    "logistic_regression": dict(preprocessing="StandardScaler", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=400, training_rows=150_000),
    "gaussian_nb": dict(var_smoothing=1e-8, preprocessing="none (scale-invariant)"),
    "decision_tree": dict(criterion="gini", max_depth=12, min_samples_leaf=50, class_weight="balanced"),
    "random_forest": dict(n_estimators=150, max_depth=24, min_samples_leaf=10, max_samples=0.3, max_features="sqrt", criterion="gini"),
    "lightgbm": dict(n_estimators=120, learning_rate=0.08, num_leaves=31, class_weight="balanced", objective="binary log loss"),
    "bagging": dict(estimator="DecisionTree(depth 14, min_leaf 20, balanced)", n_estimators=20, max_samples=0.1, bootstrap=True),
    "mlp": dict(epochs=12, h=256, depth=4, p=0.3, lr=2e-3, bs=512, wd=0.01, alpha=1.0, loss="BCE + listwise softmax", preprocessing="StandardScaler"),
    "knn": dict(k=50, weights="1/(euclidean distance + 1e-3)", metric="euclidean on standardised features", backend="GPU brute force"),
}
BALANCED = {"logistic_regression", "decision_tree", "lightgbm", "bagging"}      # class_weight='balanced' -> probabilities need prior correction


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fit_model(name, fit_df, seed):
    """-> (predict(df) -> positive probability, extra dict). Standardisation (where used) is fit on fit_df only."""
    X, y = fit_df[FE].to_numpy(np.float32), fit_df.label.to_numpy(); extra = {}
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        idx = np.random.RandomState(seed).choice(len(X), min(150_000, len(X)), replace=False)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=400, random_state=seed)).fit(X[idx].astype(np.float64), y[idx])
        extra["coef"] = m[-1].coef_[0]; return (lambda df: m.predict_proba(df[FE].to_numpy(np.float64))[:, 1]), extra
    if name == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        m = GaussianNB(var_smoothing=1e-8).fit(X, y); return (lambda df: m.predict_proba(df[FE].to_numpy(np.float32))[:, 1]), extra
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=seed).fit(X, y)
        extra["importance"] = m.feature_importances_; return (lambda df: m.predict_proba(df[FE].to_numpy(np.float32))[:, 1]), extra
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=150, max_depth=24, min_samples_leaf=10, max_samples=0.3, max_features="sqrt", n_jobs=-1, random_state=seed).fit(X, y)
        extra["importance"] = m.feature_importances_; return (lambda df: m.predict_proba(df[FE].to_numpy(np.float32))[:, 1]), extra
    if name == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.08, num_leaves=31, class_weight="balanced", n_jobs=32, random_state=seed, verbose=-1).fit(X, y)
        extra["importance"] = m.feature_importances_; return (lambda df: m.predict_proba(df[FE].to_numpy(np.float32))[:, 1]), extra
    if name == "bagging":
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        m = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=14, min_samples_leaf=20, class_weight="balanced", random_state=seed),
                              n_estimators=20, max_samples=0.1, bootstrap=True, n_jobs=8, random_state=seed).fit(X, y)
        return (lambda df: m.predict_proba(df[FE].to_numpy(np.float32))[:, 1]), extra
    if name == "mlp":
        import torch, train_mlp as M
        mu, sd = fit_df[FE].mean(), fit_df[FE].std().replace(0, 1)
        trips = np.array(sorted(fit_df.trip_id.unique())); es = set(np.random.RandomState(seed).choice(trips, max(4, len(trips) // 8), replace=False))
        a_df, e_df = fit_df[~fit_df.trip_id.isin(es)], fit_df[fit_df.trip_id.isin(es)]
        T = lambda df: torch.tensor(((df[FE] - mu) / sd).to_numpy(np.float32), device="cuda").view(-1, K, len(FE))
        Yt = lambda df: torch.tensor(df.label.to_numpy(np.float32), device="cuda").view(-1, K)
        torch.manual_seed(seed); m = M.fit(T(a_df), Yt(a_df), T(e_df), Yt(e_df), argparse.Namespace(**{k: v for k, v in CONFIG["mlp"].items() if k in ("epochs", "h", "depth", "p", "lr", "bs", "wd", "alpha")}))
        extra["history"] = m.history; return (lambda df: M.predict(m, T(df))[0]), extra
    if name == "knn":
        import torch, train_knn
        mu, sd = fit_df[FE].mean(), fit_df[FE].std().replace(0, 1)
        Xt = torch.tensor(((fit_df[FE] - mu) / sd).to_numpy(np.float32), device="cuda"); yt = torch.tensor(y.astype(np.float32), device="cuda")
        return (lambda df: train_knn.knn_prob(Xt, yt, torch.tensor(((df[FE] - mu) / sd).to_numpy(np.float32), device="cuda"), 50)), extra
    raise ValueError(name)


def calibrated_logit(name, p, fit_labels):
    z = logit(p)
    if name in BALANCED:                                   # balanced class weights inflate the odds by ~ n_neg / n_pos
        z = z - np.log((fit_labels == 0).sum() / max((fit_labels == 1).sum(), 1))
    return z.astype(np.float32)


def cand_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred)),
                f1=float(f1_score(y, pred)), macro_f1=float(f1_score(y, pred, average="macro")),
                log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])), top1=C.top1_accuracy(y, p), confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def route_continuity(df, p) -> dict:
    """Adjacent fixes <= 5 s apart whose top-1 arcs are identical or directly connected (same definition as the other model folders)."""
    T = df[["trip_id", "observation_id", "candidate_arc"]].copy(); ok = n = 0
    A = df.candidate_arc.to_numpy().reshape(-1, K); P = p.reshape(-1, K); k = P.argmax(1); top = A[np.arange(len(A)), k]
    ids = df.trip_id.to_numpy()[::K]
    for trip in pd.unique(ids):
        sel = ids == trip; pa = top[sel]
        raw, _, arcs, _ = load_trip("student_test", trip); track, _ = clean_track(raw, trip)
        assert len(track) == len(pa)
        short = np.diff(track[:, 2]) <= 5.0
        conn = (pa[:-1] == pa[1:]) | (arcs[pa[:-1], 1] == arcs[pa[1:], 0])
        ok += int((conn & short).sum()); n += int(short.sum())
    return dict(route_continuity=ok / n, connected=ok, eligible=n)


def training_sample(train):
    """20,000 randomly chosen whole fixes (320k candidate rows) spread over all training trips, for the in-sample score."""
    fixes = np.random.RandomState(SEED).choice(len(train) // K, min(20_000, len(train) // K), replace=False)
    return train.iloc[(fixes[:, None] * K + np.arange(K)).ravel()]


def training_score(name, pred, train):
    return float("nan") if name == "knn" else float(f1_score(training_sample(train).label, pred(training_sample(train)) >= 0.5, average="macro"))


def run(name, d):
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    train = d[(d.split == "train") & d.fit_ok].reset_index(drop=True); te = d[d.split == "student_test"].reset_index(drop=True)
    fold = train.trip_id.map(C.fold_assignment()).to_numpy()
    rows, oof_z = [], np.zeros(len(train), np.float32)
    for f in range(1, 6):
        tr, va = train[fold != f], train[fold == f]
        t0 = time.time(); pred, extra = fit_model(name, tr, SEED + f); fit_s = time.time() - t0
        pv = pred(va); oof_z[fold == f] = calibrated_logit(name, pv, tr.label.to_numpy())
        m = cand_metrics(va.label.to_numpy(), pv); m.pop("confusion_matrix")
        rows.append(dict(fold=f, training_trips=int(tr.trip_id.nunique()), validation_trips=int(va.trip_id.nunique()), training_candidates=len(tr), validation_candidates=len(va),
                         validation_macro_f1=m["macro_f1"], validation_log_loss=m["log_loss"], validation_top1=m["top1"], fit_seconds=fit_s))
        print(f"{name} CV fold {f}/5: macro-F1={m['macro_f1']:.4f} top1={m['top1']:.4f} fit={fit_s:.1f}s", flush=True)
    cv = pd.DataFrame(rows); cv.to_csv(out / "cv_fold_metrics.csv", index=False)
    t0 = time.time(); pred, extra = fit_model(name, train, SEED); fit_s = time.time() - t0
    times = []
    for _ in range(3):
        t0 = time.time(); pt = pred(te); times.append(time.time() - t0)
    y = te.label.to_numpy(); m = cand_metrics(y, pt); m.update(route_continuity(te, pt))
    has = te.groupby(["trip_id", "observation_id"]).label.max()
    m["candidate_miss_rate"] = float(1 - has.mean())
    train_f1 = training_score(name, pred, train)
    res = dict(model=TITLE[name], key=name, family=FAMILY[name], configuration=CONFIG[name], random_state=SEED, features=FE, threshold=0.5,
               cv_macro_f1_mean=float(cv.validation_macro_f1.mean()), cv_macro_f1_std=float(cv.validation_macro_f1.std(ddof=1)), cv_top1_mean=float(cv.validation_top1.mean()),
               cv_log_loss_mean=float(cv.validation_log_loss.mean()), final_training_macro_f1=train_f1,
               final_fit_seconds=fit_s, student_test=m, student_test_inference_seconds=float(np.mean(times)), student_test_inference_seconds_all=[float(t) for t in times],
               student_test_candidates=len(te), student_test_observations=int(len(te) // K), cv_total_fit_seconds=float(cv.fit_seconds.sum()))
    (out / "metrics.json").write_text(json.dumps(res, indent=2)); (out / "config.json").write_text(json.dumps(dict(CONFIG[name], features=FE, random_state=SEED), indent=2))
    # top-1 predictions per fix (compact) + scores for the HMM stage
    A = te.candidate_arc.to_numpy().reshape(-1, K); P = pt.reshape(-1, K); k = P.argmax(1); Y = y.reshape(-1, K)
    ids = te[["trip_id", "observation_id"]].iloc[::K].reset_index(drop=True)
    pd.DataFrame(dict(trip_id=ids.trip_id, observation_id=ids.observation_id, predicted_arc=A[np.arange(len(A)), k], positive_probability=P[np.arange(len(A)), k],
                      correct=Y[np.arange(len(A)), k] == 1)).to_csv(out / "student_test_top1_predictions.csv", index=False)
    pd.DataFrame(dict(trip_id=train.trip_id, observation_id=train.observation_id, candidate_arc=train.candidate_arc, label=train.label, z_cal=oof_z)).to_parquet(CACHE / f"oof_{name}.parquet")
    pd.DataFrame(dict(trip_id=te.trip_id, observation_id=te.observation_id, candidate_arc=te.candidate_arc, label=te.label,
                      z_cal=calibrated_logit(name, pt, train.label.to_numpy()))).to_parquet(CACHE / f"test_{name}.parquet")
    d_ = ConfusionMatrixDisplay(np.array(m["confusion_matrix"]), display_labels=["Non-reference", "Reference"]); d_.plot(cmap="Blues", values_format="d", colorbar=False)
    d_.ax_.set_title(f"{TITLE[name]}: student-test candidate confusion matrix"); d_.figure_.tight_layout(); d_.figure_.savefig(out / "confusion_matrix.png", dpi=150); plt.close(d_.figure_)
    if "importance" in extra:
        pd.DataFrame(dict(feature=FE, importance=extra["importance"])).sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    if "coef" in extra:
        pd.DataFrame(dict(feature=FE, standardised_coefficient=extra["coef"])).sort_values("standardised_coefficient", key=abs, ascending=False).to_csv(out / "feature_coefficients.csv", index=False)
    if "history" in extra:
        h = pd.DataFrame(extra["history"]); h.to_csv(out / "training_history.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); ax[0].plot(h.epoch, h.train_loss, marker="o"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss (BCE + listwise CE)"); ax[0].set_title("MLP training loss")
        ax[1].plot(h.epoch, h.val_logloss, marker="o", color="#E15759"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation log loss"); a2 = ax[1].twinx(); a2.plot(h.epoch, h.val_top1, marker="s", color="#59A14F"); a2.set_ylabel("validation Top-1")
        ax[1].set_title("MLP validation (early-stopping trips)"); fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)
    print(f"{name} STUDENT-TEST: macro-F1={m['macro_f1']:.4f} top1={m['top1']:.4f} logloss={m['log_loss']:.4f} continuity={m['route_continuity']:.4f}", flush=True)


def add_training_scores(names, d):
    """Fill `final_training_macro_f1` of existing results (refits the final model with the same seed; no CV, no re-evaluation)."""
    train = d[(d.split == "train") & d.fit_ok].reset_index(drop=True)
    for name in names:
        p = OUT / name / "metrics.json"
        if not p.exists(): continue
        m = json.loads(p.read_text()); pred, _ = fit_model(name, train, SEED); m["final_training_macro_f1"] = training_score(name, pred, train); m.pop("cv_training_macro_f1_mean", None)
        p.write_text(json.dumps(m, indent=2)); cvp = OUT / name / "cv_fold_metrics.csv"; cv = pd.read_csv(cvp)
        if "training_macro_f1" in cv: cv.drop(columns="training_macro_f1").to_csv(cvp, index=False)
        print(f"{name}: in-sample Macro F1 {m['final_training_macro_f1']:.4f} (CV {m['cv_macro_f1_mean']:.4f}, student-test {m['student_test']['macro_f1']:.4f})", flush=True)


def remeasure_inference(names, d):
    """Re-time student-test inference of existing results on an otherwise idle machine (3 passes, mean)."""
    train = d[(d.split == "train") & d.fit_ok].reset_index(drop=True); te = d[d.split == "student_test"].reset_index(drop=True)
    for name in names:
        p = OUT / name / "metrics.json"
        if not p.exists(): continue
        m = json.loads(p.read_text()); pred, _ = fit_model(name, train, SEED); times = []
        for _ in range(3):
            t0 = time.time(); pred(te); times.append(time.time() - t0)
        m["student_test_inference_seconds_first_run_under_load"] = m["student_test_inference_seconds"]
        m["student_test_inference_seconds"], m["student_test_inference_seconds_all"] = float(np.mean(times)), [float(t) for t in times]
        p.write_text(json.dumps(m, indent=2)); print(f"{name}: inference {np.mean(times):.3f} s on an idle machine (was {m['student_test_inference_seconds_first_run_under_load']:.3f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="*", default=MODELS); ap.add_argument("--time-only", action="store_true", help="re-measure inference time of existing results"); ap.add_argument("--train-score-only", action="store_true", help="only (re)compute the in-sample training Macro F1 of existing results")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    d = C.build_frame()
    if a.train_score_only:
        return add_training_scores(a.models, d)
    if a.time_only:
        return remeasure_inference(a.models, d)
    for name in a.models:
        t0 = time.time(); run(name, d); print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

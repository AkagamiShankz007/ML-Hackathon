"""Bayesian hyperparameter optimisation (Optuna TPE) of the final Logistic Regression
(StandardScaler [-> polynomial features] -> LogisticRegression/lbfgs on the 18 corrected v2 candidate features).

Objective : maximise mean candidate-level Macro-F1 (threshold 0.5) over 5 trip-grouped CV folds on the 80 training trips
            (same folds as the other model-selection experiments). Observation Top-1, precision, recall and calibrated
            log loss are recorded per trial as secondary measures.
Search    : inverse regularisation C, positive-class weight, training-sample size, polynomial feature degree (1 or 2).
Untuned   : the team's default configuration (C=1, balanced class weights, 150k rows, no polynomial terms) refitted on the same
            features, folds and metrics, so tuned vs untuned differ only in hyperparameters. Trips with corrupt labels are excluded.
Outputs   : artifacts/bayesian_optimization/ (trials, best params, plots, tuned-vs-untuned comparison)
            artifacts/cache/ (out-of-fold + student-test scores consumed by src/hmm_viterbi.py, tuned pipeline)
Usage     : python src/bayesian_opt_logistic.py [--trials 60]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, optuna
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logreg_common as C

OUT = C.ROOT / "artifacts" / "bayesian_optimization"; CACHE = C.ROOT / "artifacts" / "cache"
SPACE = {
    "C": "log-uniform [1e-4, 1e2]",
    "pos_weight": "log-uniform [1, 30]  (class_weight {0: 1, 1: pos_weight}; 'balanced' on this data is about 16.5)",
    "n_rows": [150_000, 300_000],
    "poly_degree": [1, 2],
    "fixed": dict(solver="lbfgs", penalty="l2", max_iter=400, random_state=C.SEED, features=C.FEATURES),
}
BASELINE = dict(C=1.0, pos_weight=16.5, n_rows=150_000, poly_degree=1)    # team's default configuration (balanced class weights ~ 16.5 on this data)


def fit_lr(X, y, cfg, seed):
    idx = np.random.RandomState(seed).choice(len(X), min(cfg["n_rows"], len(X)), replace=False)
    steps = [StandardScaler()]
    if cfg["poly_degree"] > 1:
        steps += [PolynomialFeatures(cfg["poly_degree"], include_bias=False), StandardScaler()]
    pipe = make_pipeline(*steps, LogisticRegression(C=cfg["C"], class_weight={0: 1.0, 1: cfg["pos_weight"]}, solver="lbfgs",
                                                    max_iter=400, random_state=C.SEED))
    return pipe.fit(X[idx], y[idx])


def metrics(y, z, pos_weight):
    pred = (z >= 0).astype(int)                      # logit >= 0  <=>  class-weighted probability >= 0.5
    p = 1 / (1 + np.exp(-(z - np.log(pos_weight))))  # prior-corrected (calibrated) probability
    return dict(macro_f1=float(f1_score(y, pred, average="macro")), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred)), log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), top1=C.top1_accuracy(y, z))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=60); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    d = C.build_frame(); train = d[(d.split == "train") & d.fit_ok].reset_index(drop=True); te = d[d.split == "student_test"].reset_index(drop=True)
    fold = train.trip_id.map(C.fold_assignment()).to_numpy()
    X, y = train[C.FEATURES].to_numpy(np.float64), train.label.to_numpy(); Xt, yt = te[C.FEATURES].to_numpy(np.float64), te.label.to_numpy()
    print(f"{len(train):,} train / {len(te):,} test candidates", flush=True)

    def cv(cfg):
        def one(f):
            tr, va = fold != f, fold == f
            z = fit_lr(X[tr], y[tr], cfg, C.SEED + f).decision_function(X[va])
            return f, z, metrics(y[va], z, cfg["pos_weight"])
        return Parallel(n_jobs=5, backend="threading")(delayed(one)(f) for f in range(1, 6))

    def objective(trial):
        cfg = dict(C=trial.suggest_float("C", 1e-4, 1e2, log=True), pos_weight=trial.suggest_float("pos_weight", 1.0, 30.0, log=True),
                   n_rows=trial.suggest_categorical("n_rows", SPACE["n_rows"]), poly_degree=trial.suggest_categorical("poly_degree", SPACE["poly_degree"]))
        t0 = time.time(); m = [r[2] for r in cv(cfg)]
        for k in ("macro_f1", "top1", "log_loss", "precision", "recall"):
            trial.set_user_attr(f"cv_{k}", float(np.mean([x[k] for x in m])))
        trial.set_user_attr("cv_macro_f1_std", float(np.std([x["macro_f1"] for x in m], ddof=1))); trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number:>2}: F1={trial.user_attrs['cv_macro_f1']:.4f} top1={trial.user_attrs['cv_top1']:.4f} {cfg} ({time.time() - t0:.0f}s)", flush=True)
        return trial.user_attrs["cv_macro_f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=10))
    study.enqueue_trial(BASELINE)                                   # trial 0 = the team's configuration
    optuna.logging.set_verbosity(optuna.logging.WARNING); study.optimize(objective, n_trials=a.trials)
    best = dict(study.best_params); print("BEST", best, study.best_value, flush=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")); trials.to_csv(OUT / "trials.csv", index=False)

    # ---- untuned = the team's default configuration, tuned = Bayesian best; both refit on all clean training trips
    rows = []
    t0 = time.time(); pipe_b = fit_lr(X, y, BASELINE, C.SEED); fit_b = time.time() - t0
    t0 = time.time(); z_b = pipe_b.decision_function(Xt); inf_b = time.time() - t0
    res_b = cv(BASELINE); mb = [r[2] for r in res_b]; m = metrics(yt, z_b, BASELINE["pos_weight"])
    rows.append(dict(model="untuned (team's default config)", **BASELINE, cv_macro_f1=np.mean([x["macro_f1"] for x in mb]),
                     cv_macro_f1_std=np.std([x["macro_f1"] for x in mb], ddof=1), cv_top1=np.mean([x["top1"] for x in mb]),
                     test_macro_f1=m["macro_f1"], test_precision=m["precision"], test_recall=m["recall"], test_top1=m["top1"],
                     test_log_loss=m["log_loss"], fit_seconds=fit_b, inference_seconds=inf_b))
    t0 = time.time(); pipe = fit_lr(X, y, best, C.SEED); fit_s = time.time() - t0
    t0 = time.time(); z = pipe.decision_function(Xt); inf_s = time.time() - t0
    res = cv(best); mt = [r[2] for r in res]; m = metrics(yt, z, best["pos_weight"])
    rows.append(dict(model="tuned (Bayesian best)", **best, cv_macro_f1=np.mean([x["macro_f1"] for x in mt]), cv_macro_f1_std=np.std([x["macro_f1"] for x in mt], ddof=1),
                     cv_top1=np.mean([x["top1"] for x in mt]), test_macro_f1=m["macro_f1"], test_precision=m["precision"], test_recall=m["recall"],
                     test_top1=m["top1"], test_log_loss=m["log_loss"], fit_seconds=fit_s, inference_seconds=inf_s))
    cmp = pd.DataFrame(rows); cmp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)

    # ---- scores for the HMM stage: out-of-fold (train) and student-test, prior-corrected logit z_cal = z - log(pos_weight)
    oof = train[["trip_id", "observation_id", "candidate_arc", "label"]].copy(); oof["z_cal"] = 0.0
    for f, z_, _ in res: oof.loc[fold == f, "z_cal"] = z_ - np.log(best["pos_weight"])
    oof["z_cal"] = oof["z_cal"].astype(np.float32); oof.to_parquet(CACHE / "oof_logreg.parquet")
    tt = te[["trip_id", "observation_id", "candidate_arc", "label"]].copy(); tt["z_cal"] = (z - np.log(best["pos_weight"])).astype(np.float32)
    tt.to_parquet(CACHE / "test_logreg.parquet")
    joblib.dump(dict(pipeline=pipe, cfg=best, features=C.FEATURES), CACHE / "tuned_model.joblib")
    (OUT / "best_params.json").write_text(json.dumps(dict(
        best_params=best, best_cv_macro_f1=study.best_value, n_trials=len(study.trials), baseline_config=BASELINE, search_space=SPACE,
        sampler="TPE (Optuna), seed 42, 10 random start-up trials",
        objective="mean 5-fold trip-grouped CV candidate Macro-F1 at threshold 0.5",
        cv="StratifiedGroupKFold(5, shuffle, seed 42) grouped by trip on the 80 train trips"), indent=2))
    # ---- plots
    v = trials[trials.state == "COMPLETE"]; fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(v.number, v.value, "o", alpha=.6, label="trial"); ax[0].plot(v.number, v.value.cummax(), "r-", label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV Macro-F1"); ax[0].set_title("Optimisation history"); ax[0].legend()
    for deg, g in v.groupby("params_poly_degree"): ax[1].scatter(g.params_C, g.value, label=f"poly degree {deg}", alpha=.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("C"); ax[1].set_ylabel("CV Macro-F1"); ax[1].set_title("C vs score"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "optimization_history.png", dpi=150); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study); fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1]); ax.set_title("Hyper-parameter importance (fANOVA)"); fig.tight_layout()
        fig.savefig(OUT / "param_importance.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("importance plot skipped:", e)
    print(cmp.T.to_string())


if __name__ == "__main__":
    main()

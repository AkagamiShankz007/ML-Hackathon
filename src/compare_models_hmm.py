"""Does a stronger classifier still win after HMM/Viterbi decoding?  (final-model decision experiment)

For every model of src/model_comparison.py (all on the same 18 corrected v2 features, src/candidates_v2.py) this script
  1. produces out-of-fold scores for the 80 training trips with the SAME trip-grouped folds as the logistic regression
     (src/logreg_common.fold_assignment) and student-test scores from a model fitted on all 80 trips,
  2. tunes the HMM decoder parameters on the out-of-fold scores only (same search space/trials as src/hmm_tuning.py),
  3. evaluates classifier-only vs classifier+HMM on the ten student-test trips.
The logistic-regression row is read from artifacts/hmm/comparison.csv (Bayesian-tuned).
Outputs: artifacts/model_comparison/ (hmm_comparison.csv, hmm_decoder_params.json). Usage: python src/compare_models_hmm.py   (after src/model_comparison.py)"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, optuna, torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "artifacts/model_selection/mlp"):
    sys.path.insert(0, str(ROOT / p))
import logreg_common as C
import train_mlp as M
import hmm_tuning as HV
from hmm_decoder import DEFAULT

OUT = ROOT / "artifacts" / "model_comparison"; CACHE = ROOT / "artifacts" / "cache"
K, FE, SEED = C.K, C.FEATURES, C.SEED
MLP_CFG = dict(epochs=12, h=256, depth=4, p=0.3, lr=2e-3, bs=512, wd=0.01, alpha=1.0)   # as in artifacts/model_selection/mlp


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).astype(np.float32)


def fit_score(name, fit_df, ev_list, seed):
    """-> list of calibrated logits (one array per frame in ev_list)."""
    trips = np.array(sorted(fit_df.trip_id.unique()))
    es = set(np.random.RandomState(seed).choice(trips, max(4, len(trips) // 8), replace=False))
    a_df, e_df = fit_df[~fit_df.trip_id.isin(es)], fit_df[fit_df.trip_id.isin(es)]
    if name == "mlp":
        mu = a_df[FE].mean(); sd = a_df[FE].std().replace(0, 1)
        T = lambda df: torch.tensor(((df[FE] - mu) / sd).to_numpy(np.float32), device="cuda").view(-1, K, len(FE))
        Yt = lambda df: torch.tensor(df.label.to_numpy(np.float32), device="cuda").view(-1, K)
        torch.manual_seed(seed)
        model = M.fit(T(a_df), Yt(a_df), T(e_df), Yt(e_df), argparse.Namespace(**MLP_CFG))
        return [logit(M.predict(model, T(e))[0]) for e in ev_list]
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, reg_lambda=1.0, n_jobs=32, random_state=seed, verbose=-1)
    clf.fit(a_df[FE].to_numpy(np.float32), a_df.label, eval_set=[(e_df[FE].to_numpy(np.float32), e_df.label)],
            callbacks=[lgb.early_stopping(40, verbose=False)])
    return [logit(clf.predict_proba(e[FE].to_numpy(np.float32))[:, 1]) for e in ev_list]


def scores_for(name, train, te, fold):
    f_oof, f_te = CACHE / f"oof_{name}.parquet", CACHE / f"test_{name}.parquet"
    if f_oof.exists() and f_te.exists():
        return pd.read_parquet(f_oof), pd.read_parquet(f_te)
    oof = train[["trip_id", "observation_id", "candidate_arc", "label"]].copy(); oof["z_cal"] = np.float32(0)
    for f in range(1, 6):
        t0 = time.time(); tr, va = train[fold != f], train[fold == f]
        oof.loc[fold == f, "z_cal"] = fit_score(name, tr, [va], SEED + f)[0]
        print(f"{name} fold {f}/5 done ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time(); tt = te[["trip_id", "observation_id", "candidate_arc", "label"]].copy(); tt["z_cal"] = fit_score(name, train, [te], SEED)[0]
    print(f"{name} final fit + student-test scoring ({time.time() - t0:.0f}s)", flush=True)
    oof.to_parquet(f_oof); tt.to_parquet(f_te)
    return oof, tt


def tune_decoder(trips, n_trials):
    def objective(trial):
        p = dict(alpha=trial.suggest_float("alpha", 0.2, 3.0, log=True), same=trial.suggest_float("same", -3, 3), hop1=0.0,
                 hop2=trial.suggest_float("hop2", -8, 3), hop3=trial.suggest_float("hop3", -8, 3),
                 uturn=trial.suggest_float("uturn", -12, 2), none=trial.suggest_float("none", -15, 0))
        return HV.evaluate(trips, p)["top1_accuracy"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=15))
    study.enqueue_trial({k: v for k, v in DEFAULT.items() if k != "hop1"})
    optuna.logging.set_verbosity(optuna.logging.WARNING); study.optimize(objective, n_trials=n_trials)
    return dict(DEFAULT, **study.best_params, hop1=0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=60); ap.add_argument("--models", nargs="+", default=["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "lightgbm", "bagging", "mlp", "knn"])
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    d = C.build_frame(); train = d[(d.split == "train") & d.fit_ok].reset_index(drop=True); te = d[d.split == "student_test"].reset_index(drop=True)
    fold = train.trip_id.map(C.fold_assignment()).to_numpy()
    rows, params = [], {}
    for name in a.models:
        oof, tt = scores_for(name, train, te, fold)
        tr_trips, te_trips = HV.prepare(oof, "train"), HV.prepare(tt, "student_test")
        t0 = time.time(); best = tune_decoder(tr_trips, a.trials); params[name] = best; print(f"{name}: decoder tuned in {time.time() - t0:.0f}s {best}", flush=True)
        for part, trips in (("train (out-of-fold)", tr_trips), ("student_test", te_trips)):
            for method, p in (("classifier only", None), ("classifier + HMM (tuned)", best)):
                rows.append(dict(model=name, partition=part, method=method, **HV.evaluate(trips, p)))
    cmp = pd.DataFrame(rows)
    lg = pd.read_csv(ROOT / "artifacts" / "hmm" / "comparison.csv"); lg = lg[lg.method.isin(["classifier only", "classifier + HMM (tuned)"])].copy()
    lg.insert(0, "model", "logistic_regression_tuned")
    cmp = pd.concat([lg[cmp.columns.intersection(lg.columns)], cmp], ignore_index=True)
    prev_csv, prev_json = OUT / "hmm_comparison.csv", OUT / "hmm_decoder_params.json"
    if prev_csv.exists():                                  # keep results of models evaluated in earlier runs (re-running a model replaces its rows)
        old = pd.read_csv(prev_csv); cmp = pd.concat([old[~old.model.isin(set(cmp.model))], cmp], ignore_index=True)
        if prev_json.exists(): params = {**json.loads(prev_json.read_text()), **params}
    order = {m: i for i, m in enumerate(["logistic_regression_tuned"] + a.models + ["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "lightgbm", "bagging", "mlp", "knn"])}
    cmp = cmp.sort_values("model", key=lambda c: c.map(lambda m: order.get(m, 99)), kind="stable").reset_index(drop=True)
    cmp.to_csv(prev_csv, index=False); prev_json.write_text(json.dumps(params, indent=2))
    pd.set_option("display.width", 250)
    print(cmp[["model", "partition", "method", "top1_accuracy", "route_continuity", "route_arc_precision", "route_arc_recall", "route_arc_f1"]].round(4).to_string())


if __name__ == "__main__":
    main()

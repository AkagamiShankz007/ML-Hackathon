"""Export the final model bundle to artifacts/final_model/ (everything the API needs at inference time).

Bundle contents
  logistic_regression_final.joblib : sklearn Pipeline (StandardScaler [-> PolynomialFeatures -> StandardScaler] -> LogisticRegression)
  feature_spec.json                : candidate-feature columns, engineered-feature formulas, calibration (pos_weight)
  hmm_config.json                  : decoder parameters, relation classes, gap threshold
  model_manifest.json              : configuration, seeds, library versions, CV / student-test metrics, provenance

Run after src/candidates_v2.py, src/bayesian_opt_logistic.py and src/hmm_tuning.py:   python src/export_final_model.py"""
from __future__ import annotations
import json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, scipy, sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logreg_common as C

ROOT = C.ROOT; OUT = ROOT / "artifacts" / "final_model"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tuned = joblib.load(ROOT / "artifacts" / "cache" / "tuned_model.joblib")
    cfg = tuned["cfg"]; assert list(tuned["features"]) == C.FEATURES
    joblib.dump(tuned["pipeline"], OUT / "logistic_regression_final.joblib", compress=3)
    (OUT / "feature_spec.json").write_text(json.dumps(dict(
        candidate_generation="src/candidates_v2.py: 16 arcs with the smallest exact point-to-segment distance (<=40 m sub-segment index, 96 shortlisted)",
        model_input_order=C.FEATURES, candidates_per_observation=C.K,
        feature_notes={"bearing_delta": "|wrap(gps_bearing - arc_bearing)| in radians (pi/2 when direction unknown); one-step displacement",
                       "bearing_delta_smooth": "same, from a +-2-fix window (>=3 m displacement, no gap > 5 s)",
                       "cos_bearing / cos_bearing_smooth": "cosine of the above, 0 when direction unknown",
                       "speed_aligned / speed_aligned_smooth": "speed * cosine of the bearing difference",
                       "obs_dist_rank / rel_dist": "rank of the candidate by distance (1..16) and distance minus the closest candidate's"},
        score="decision_function(pipeline, X)   (logit of the class-weighted model)",
        calibration=dict(pos_weight=cfg["pos_weight"], calibrated_logit="decision_function - log(pos_weight)",
                         probability="sigmoid(calibrated_logit)")), indent=2))
    hmm = json.loads((ROOT / "artifacts" / "hmm" / "hmm_config.json").read_text())
    (OUT / "hmm_config.json").write_text(json.dumps(hmm, indent=2))
    bo = json.loads((ROOT / "artifacts" / "bayesian_optimization" / "best_params.json").read_text())
    tv = pd.read_csv(ROOT / "artifacts" / "bayesian_optimization" / "tuned_vs_untuned.csv")
    cmp = pd.read_csv(ROOT / "artifacts" / "hmm" / "comparison.csv")
    row = lambda part, method: cmp[(cmp.partition == part) & (cmp.method == method)].iloc[0]
    st = lambda method: {k: float(row("student_test", method)[k]) for k in
                         ("top1_accuracy", "route_continuity", "route_arc_precision", "route_arc_recall", "route_arc_f1")}
    t = tv[tv.model.str.startswith("tuned")].iloc[0]
    (OUT / "model_manifest.json").write_text(json.dumps(dict(
        model_name="logistic_regression_bayes_tuned+hmm", created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        classifier=dict(family="Logistic Regression (Bayesian-tuned, corrected v2 features)", hyperparameters=cfg,
                        fixed=bo["search_space"]["fixed"], random_seed=C.SEED,
                        trained_on="random sample of candidate rows of the clean training trips (label-quality filter: median reference distance <= 50 m)",
                        optimisation=dict(objective=bo["objective"], cv=bo["cv"], n_trials=bo["n_trials"], sampler=bo["sampler"],
                                          best_cv_macro_f1=bo["best_cv_macro_f1"])),
        decoder=dict(type="HMM / Viterbi over the 16 candidates per GPS fix", params=hmm["decoder_params"], tuned_on=hmm["tuned_on"]),
        metrics=dict(classifier_tuned_student_test=dict(macro_f1=float(t.test_macro_f1), top1_accuracy=float(t.test_top1),
                                                        log_loss=float(t.test_log_loss), precision=float(t.test_precision), recall=float(t.test_recall)),
                     student_test_classifier_only=st("classifier only"), student_test_with_hmm=st("classifier + HMM (tuned)")),
        library_versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                              scikit_learn=sklearn.__version__, pandas=pd.__version__),
        notes=["Reference routes are never used at inference time; they only produce the evaluation metrics inside the API.",
               "z_cal = decision_function - log(pos_weight) is the emission input of the HMM."]), indent=2))
    print("exported:", *sorted(p.name for p in OUT.iterdir()), sep="\n  ")


if __name__ == "__main__":
    main()

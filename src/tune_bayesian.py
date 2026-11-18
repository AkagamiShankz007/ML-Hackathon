import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

# Disable optuna verbosity for clean logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
STUDENT_TEST_DIR = ROOT / "Student folder" / "student_test"

FEATURE_COLUMNS = [
    "distance_m",
    "log_distance",
    "projection",
    "segment_length_m",
    "log_segment_length",
    "bearing_delta",
    "gps_speed_mps",
    "log_gps_gap_s",
    "motion_valid",
]


def load_student_test_arcs() -> dict[str, np.ndarray]:
    arcs_dict = {}
    for trip_dir in STUDENT_TEST_DIR.iterdir():
        if trip_dir.is_dir():
            trip_id = trip_dir.name
            arcs_path = trip_dir / f"{trip_id}.arcs"
            if arcs_path.exists():
                arcs_dict[trip_id] = np.loadtxt(arcs_path, dtype=np.int64, ndmin=2)
    return arcs_dict


def compute_top1_accuracy_and_continuity(
    df_eval: pd.DataFrame, probs: np.ndarray, arcs_dict: dict[str, np.ndarray]
) -> tuple[float, float]:
    temp_df = df_eval[["trip_id", "observation_id", "candidate_arc", "reference_arc"]].copy()
    temp_df["prob"] = probs

    idx = temp_df.groupby(["trip_id", "observation_id"])["prob"].idxmax()
    top1_df = temp_df.loc[idx].sort_values(["trip_id", "observation_id"])

    is_correct = top1_df["candidate_arc"] == top1_df["reference_arc"]
    top1_accuracy = float(is_correct.mean())

    total_transitions = 0
    connected_transitions = 0
    for trip_id, trip_obs in top1_df.groupby("trip_id"):
        arcs = arcs_dict.get(trip_id)
        if arcs is None:
            continue
        predicted_arcs = trip_obs["candidate_arc"].to_numpy(dtype=np.int64)
        if len(predicted_arcs) <= 1:
            continue

        valid = (predicted_arcs >= 0) & (predicted_arcs < len(arcs))
        valid_pairs = valid[:-1] & valid[1:]
        pair_idx = np.flatnonzero(valid_pairs)
        if pair_idx.size > 0:
            connected = int(np.sum(
                arcs[predicted_arcs[pair_idx], 1] == arcs[predicted_arcs[pair_idx + 1], 0]
            ))
            total_transitions += len(predicted_arcs) - 1
            connected_transitions += connected

    continuity = float(connected_transitions / total_transitions) if total_transitions > 0 else 0.0
    return top1_accuracy, continuity


def run_bayesian_optimization():
    print("=" * 80)
    print("PATHSNAP: BAYESIAN HYPERPARAMETER OPTIMISATION (LIGHTGBM)")
    print("=" * 80)

    # 1. Load Data
    parquet_path = ARTIFACTS / "candidate_features.parquet"
    print(f"\n[1/4] Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "student_test"].reset_index(drop=True)

    X_train = df_train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = df_train["label"].to_numpy(dtype=np.int8)
    groups_train = df_train["trip_id"].to_numpy()

    X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.int8)

    student_test_arcs = load_student_test_arcs()

    # 2. Objective Function with 3-Fold GroupKFold
    gkf = GroupKFold(n_splits=3)
    cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_f1s = []
        for tr_idx, val_idx in cv_splits:
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            clf = LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            val_probs = clf.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(np.int8)
            f1 = f1_score(y_val, val_preds, average="macro")
            fold_f1s.append(f1)

        mean_f1 = float(np.mean(fold_f1s))
        trial_records.append({
            "trial_number": trial.number,
            "params": trial.params,
            "cv_macro_f1": mean_f1,
        })
        print(f"  Trial {trial.number:2d}: CV Macro F1 = {mean_f1:.4f} (leaves={params['num_leaves']}, lr={params['learning_rate']:.4f}, depth={params['max_depth']})")
        return mean_f1

    print("\n[2/4] Running 15 Bayesian Search Trials with Optuna...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t_opt_start = time.time()
    study.optimize(objective, n_trials=15)
    opt_time = time.time() - t_opt_start

    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"\nOptimization completed in {opt_time:.1f}s")
    print(f"Best Trial #{study.best_trial.number}: CV Macro F1 = {best_cv_score:.4f}")
    print(f"Best Hyperparameters: {json.dumps(best_params, indent=2)}")

    # 3. Train Final Tuned Model on 80 Trips & Evaluate on 10 Student-Test Trips
    print("\n[3/4] Refitting Tuned Model on all 80 Training Trips...")
    tuned_model = LGBMClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    t_train_start = time.time()
    tuned_model.fit(X_train, y_train)
    tuned_train_time = time.time() - t_train_start

    t_inf_start = time.time()
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]
    tuned_inf_time = time.time() - t_inf_start
    tuned_preds = (tuned_probs >= 0.5).astype(np.int8)

    tuned_macro_f1 = float(f1_score(y_test, tuned_preds, average="macro"))
    tuned_logloss = float(log_loss(y_test, tuned_probs))
    tuned_top1, tuned_continuity = compute_top1_accuracy_and_continuity(
        df_test, tuned_probs, student_test_arcs
    )

    # Load Untuned LightGBM results from prior benchmark for direct comparison
    with open(MODELS_DIR / "all_models_metrics.json", "r") as f:
        all_metrics = json.load(f)
    untuned_entry = next(m for m in all_metrics if m["slug"] == "lightgbm")

    print("\n" + "=" * 80)
    print("TUNED VS UNTUNED MODEL PERFORMANCE COMPARISON (GUIDELINES § 4)")
    print("=" * 80)
    comparison_table = [
        {
            "Metric": "Cross-validation Macro F1-score",
            "Untuned LightGBM": f"{untuned_entry['cv_macro_f1']:.4f}",
            "Tuned LightGBM": f"{best_cv_score:.4f}",
            "Delta": f"{best_cv_score - untuned_entry['cv_macro_f1']:+.4f}",
        },
        {
            "Metric": "Student-testing Top-1 Accuracy",
            "Untuned LightGBM": f"{untuned_entry['student_test_top1_accuracy']:.4f} ({untuned_entry['student_test_top1_accuracy']*100:.2f}%)",
            "Tuned LightGBM": f"{tuned_top1:.4f} ({tuned_top1*100:.2f}%)",
            "Delta": f"{tuned_top1 - untuned_entry['student_test_top1_accuracy']:+.4f}",
        },
        {
            "Metric": "Student-testing Macro F1-score",
            "Untuned LightGBM": f"{untuned_entry['student_test_macro_f1']:.4f}",
            "Tuned LightGBM": f"{tuned_macro_f1:.4f}",
            "Delta": f"{tuned_macro_f1 - untuned_entry['student_test_macro_f1']:+.4f}",
        },
        {
            "Metric": "Route Continuity",
            "Untuned LightGBM": f"{untuned_entry['student_test_route_continuity']:.4f} ({untuned_entry['student_test_route_continuity']*100:.2f}%)",
            "Tuned LightGBM": f"{tuned_continuity:.4f} ({tuned_continuity*100:.2f}%)",
            "Delta": f"{tuned_continuity - untuned_entry['student_test_route_continuity']:+.4f}",
        },
        {
            "Metric": "Student-testing Log Loss",
            "Untuned LightGBM": f"{untuned_entry['student_test_log_loss']:.4f}",
            "Tuned LightGBM": f"{tuned_logloss:.4f}",
            "Delta": f"{tuned_logloss - untuned_entry['student_test_log_loss']:+.4f}",
        },
        {
            "Metric": "Inference Time",
            "Untuned LightGBM": f"{untuned_entry['inference_time_s']:.4f} s",
            "Tuned LightGBM": f"{tuned_inf_time:.4f} s",
            "Delta": f"{tuned_inf_time - untuned_entry['inference_time_s']:+.4f} s",
        },
    ]
    comp_df = pd.DataFrame(comparison_table)
    print(comp_df.to_string(index=False))

    # 4. Save Tuned Artifacts
    print("\n[4/4] Saving tuned model weights, search history, and figures...")
    joblib.dump(tuned_model, MODELS_DIR / "lightgbm_tuned.joblib")
    comp_df.to_csv(MODELS_DIR / "tuned_vs_untuned_comparison.csv", index=False)

    tuning_payload = {
        "best_trial": study.best_trial.number,
        "best_cv_macro_f1": best_cv_score,
        "best_hyperparameters": best_params,
        "optimization_time_seconds": opt_time,
        "n_trials": len(study.trials),
        "trials": trial_records,
        "tuned_test_metrics": {
            "top1_accuracy": tuned_top1,
            "route_continuity": tuned_continuity,
            "macro_f1": tuned_macro_f1,
            "log_loss": tuned_logloss,
            "inference_time_s": tuned_inf_time,
            "training_time_s": tuned_train_time,
        },
    }
    with open(MODELS_DIR / "bayesian_tuning_results.json", "w") as f:
        json.dump(tuning_payload, f, indent=2)

    # Plot Optimization History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    trial_scores = [t["cv_macro_f1"] for t in trial_records]
    best_cum = np.maximum.accumulate(trial_scores)
    axes[0].plot(range(len(trial_scores)), trial_scores, "o-", color="#1f77b4", label="Trial Score")
    axes[0].plot(range(len(trial_scores)), best_cum, "r--", linewidth=2, label="Best Cumulative Score")
    axes[0].set_title("Optuna Bayesian Optimization History", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Trial Number")
    axes[0].set_ylabel("3-Fold CV Macro F1")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Bar comparison
    categories = ["Top-1 Acc", "Macro F1", "Log Loss"]
    un_vals = [
        untuned_entry['student_test_top1_accuracy'],
        untuned_entry['student_test_macro_f1'],
        untuned_entry['student_test_log_loss'],
    ]
    tu_vals = [tuned_top1, tuned_macro_f1, tuned_logloss]
    x = np.arange(len(categories))
    w = 0.35
    axes[1].bar(x - w/2, un_vals, w, label="Untuned LightGBM", color="#aec7e8")
    axes[1].bar(x + w/2, tu_vals, w, label="Tuned LightGBM", color="#1f77b4")
    axes[1].set_title("Untuned vs Tuned LightGBM (Student Test)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    for i in range(len(categories)):
        axes[1].text(x[i] - w/2, un_vals[i] + 0.01, f"{un_vals[i]:.3f}", ha="center", fontsize=9)
        axes[1].text(x[i] + w/2, tu_vals[i] + 0.01, f"{tu_vals[i]:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    opt_plot_path = MODELS_DIR / "bayesian_optimization_eval.png"
    plt.savefig(opt_plot_path, dpi=150)
    plt.close()
    print(f"Tuning plot saved to: {opt_plot_path}")
    print("\nBayesian Optimization successfully completed!")


if __name__ == "__main__":
    run_bayesian_optimization()

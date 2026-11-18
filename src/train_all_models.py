import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
) -> tuple[float, float, dict]:
    temp_df = df_eval[["trip_id", "observation_id", "candidate_arc", "reference_arc"]].copy()
    temp_df["prob"] = probs

    # Observation-level Top-1: pick candidate with highest probability
    idx = temp_df.groupby(["trip_id", "observation_id"])["prob"].idxmax()
    top1_df = temp_df.loc[idx].sort_values(["trip_id", "observation_id"])

    is_correct = top1_df["candidate_arc"] == top1_df["reference_arc"]
    top1_accuracy = float(is_correct.mean())

    # Sequence-level Route Continuity across trips
    total_transitions = 0
    connected_transitions = 0
    per_trip_continuity = {}

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
            n_trans = len(predicted_arcs) - 1
            total_transitions += n_trans
            connected_transitions += connected
            per_trip_continuity[trip_id] = float(connected / n_trans) if n_trans > 0 else 1.0

    route_continuity = float(connected_transitions / total_transitions) if total_transitions > 0 else 0.0

    return top1_accuracy, route_continuity, per_trip_continuity


def get_model_specs():
    specs = [
        {
            "name": "Logistic Regression",
            "slug": "logistic_regression",
            "family": "Linear and probabilistic",
            "key_config": "StandardScaler, C=1.0, balanced, lbfgs",
            "builder": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42))
            ]),
        },
        {
            "name": "Gaussian Naive Bayes",
            "slug": "gaussian_nb",
            "family": "Probabilistic",
            "key_config": "var_smoothing=1e-8",
            "builder": lambda: GaussianNB(var_smoothing=1e-8),
        },
        {
            "name": "Decision Tree",
            "slug": "decision_tree",
            "family": "Tree-based",
            "key_config": "max_depth=12, min_samples_leaf=50, balanced",
            "builder": lambda: DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=42
            ),
        },
        {
            "name": "Random Forest",
            "slug": "random_forest",
            "family": "Ensemble",
            "key_config": "n_estimators=50, max_depth=12, max_samples=0.25, balanced",
            "builder": lambda: RandomForestClassifier(
                n_estimators=50,
                max_depth=12,
                max_samples=0.25,
                min_samples_leaf=50,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        },
        {
            "name": "LightGBM",
            "slug": "lightgbm",
            "family": "Boosting",
            "key_config": "n_estimators=120, lr=0.08, num_leaves=31, balanced",
            "builder": lambda: LGBMClassifier(
                n_estimators=120,
                learning_rate=0.08,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        },
    ]
    return specs


def run_benchmark():
    print("=" * 80)
    print("PATHSNAP: BENCHMARKING 5 DISTINCT MODEL FAMILIES")
    print("=" * 80)

    # 1. Load Parquet Data
    parquet_path = ARTIFACTS / "candidate_features.parquet"
    print(f"\n[1/4] Loading preprocessed candidate features from {parquet_path}...")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} candidate rows in {time.time() - t0:.2f}s")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "student_test"].reset_index(drop=True)

    X_train = df_train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = df_train["label"].to_numpy(dtype=np.int8)
    groups_train = df_train["trip_id"].to_numpy()

    X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.int8)

    print(f"Training split: {len(df_train):,} rows ({df_train['trip_id'].nunique()} trips)")
    print(f"Student-test split: {len(df_test):,} rows ({df_test['trip_id'].nunique()} trips)")

    student_test_arcs = load_student_test_arcs()
    print(f"Loaded road network arcs for {len(student_test_arcs)} student-test trips.")

    # 2. Iterate Models
    specs = get_model_specs()
    all_results = []
    confusion_matrices = {}

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train, y_train, groups=groups_train))

    print("\n[2/4] Beginning 5-Fold GroupKFold CV and Full Training for each model...")

    for i, spec in enumerate(specs, 1):
        m_name = spec["name"]
        m_slug = spec["slug"]
        m_family = spec["family"]
        m_config = spec["key_config"]

        print(f"\n--- Model {i}/5: {m_name} ({m_family}) ---")
        print(f"Configuration: {m_config}")

        # GroupKFold Cross-Validation
        cv_macro_f1s = []
        cv_top1_accs = []
        cv_loglosses = []

        t_cv_start = time.time()
        for fold, (tr_idx, val_idx) in enumerate(splits, 1):
            fold_model = spec["builder"]()
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            fold_model.fit(X_tr, y_tr)
            val_probs = fold_model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(np.int8)

            f1 = f1_score(y_val, val_preds, average="macro")
            ll = log_loss(y_val, val_probs)

            # Fold Top-1 Acc
            val_sub = df_train.iloc[val_idx][["trip_id", "observation_id", "candidate_arc", "reference_arc"]].copy()
            val_sub["prob"] = val_probs
            top_idx = val_sub.groupby(["trip_id", "observation_id"])["prob"].idxmax()
            top1 = (val_sub.loc[top_idx, "candidate_arc"] == val_sub.loc[top_idx, "reference_arc"]).mean()

            cv_macro_f1s.append(f1)
            cv_top1_accs.append(top1)
            cv_loglosses.append(ll)

        cv_time = time.time() - t_cv_start
        mean_cv_f1 = float(np.mean(cv_macro_f1s))
        std_cv_f1 = float(np.std(cv_macro_f1s))
        mean_cv_top1 = float(np.mean(cv_top1_accs))
        print(f"  5-Fold CV Macro F1: {mean_cv_f1:.4f} +/- {std_cv_f1:.4f} (CV time: {cv_time:.1f}s)")
        print(f"  5-Fold CV Top-1 Acc: {mean_cv_top1:.4f}")

        # Train on full 80 training trips
        final_model = spec["builder"]()
        t_fit_start = time.time()
        final_model.fit(X_train, y_train)
        training_time = time.time() - t_fit_start
        print(f"  Full training complete in {training_time:.2f}s")

        # Evaluate on Student-Testing
        t_inf_start = time.time()
        test_probs = final_model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t_inf_start
        test_preds = (test_probs >= 0.5).astype(np.int8)

        # Metrics
        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_macro_f1 = f1_score(y_test, test_preds, average="macro")
        test_pos_f1 = f1_score(y_test, test_preds, pos_label=1)
        test_roc_auc = roc_auc_score(y_test, test_probs)
        test_ll = log_loss(y_test, test_probs)
        cm = confusion_matrix(y_test, test_preds)
        confusion_matrices[m_name] = cm

        top1_acc, continuity, trip_continuity = compute_top1_accuracy_and_continuity(
            df_test, test_probs, student_test_arcs
        )

        print(f"  Student-Test Top-1 Accuracy:   {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"  Student-Test Route Continuity: {continuity:.4f} ({continuity*100:.2f}%)")
        print(f"  Student-Test Macro F1:         {test_macro_f1:.4f}")
        print(f"  Student-Test Log Loss:         {test_ll:.4f}")
        print(f"  Inference Time:                {inference_time:.4f}s ({len(df_test)/inference_time:,.0f} rows/s)")

        # Save model artifact
        save_path = MODELS_DIR / f"{m_slug}.joblib"
        joblib.dump(final_model, save_path)

        result_entry = {
            "model_name": m_name,
            "slug": m_slug,
            "algorithm_family": m_family,
            "key_configuration": m_config,
            "cv_macro_f1": mean_cv_f1,
            "cv_macro_f1_std": std_cv_f1,
            "cv_top1_accuracy": mean_cv_top1,
            "student_test_top1_accuracy": top1_acc,
            "student_test_route_continuity": continuity,
            "student_test_macro_f1": test_macro_f1,
            "student_test_positive_f1": test_pos_f1,
            "student_test_precision": test_prec,
            "student_test_recall": test_rec,
            "student_test_log_loss": test_ll,
            "student_test_roc_auc": test_roc_auc,
            "training_time_s": training_time,
            "inference_time_s": inference_time,
            "confusion_matrix": cm.tolist(),
        }
        all_results.append(result_entry)

    # 3. Save Summary Table & JSON
    print("\n[3/4] Compiling comparison tables and saving metrics...")
    summary_df = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Algorithm Family": r["algorithm_family"],
            "Key Configuration": r["key_configuration"],
            "CV Macro F1-score": f"{r['cv_macro_f1']:.4f} +/- {r['cv_macro_f1_std']:.4f}",
            "Top-1 Accuracy": f"{r['student_test_top1_accuracy']:.4f}",
            "Route continuity": f"{r['student_test_route_continuity']:.4f}",
            "Student-Test Macro F1-score": f"{r['student_test_macro_f1']:.4f}",
            "Log Loss": f"{r['student_test_log_loss']:.4f}",
            "Training Time (s)": f"{r['training_time_s']:.2f}",
            "Inference Time (s)": f"{r['inference_time_s']:.4f}",
        }
        for r in all_results
    ])

    summary_df.to_csv(MODELS_DIR / "model_comparison_table.csv", index=False)
    with open(MODELS_DIR / "all_models_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 90)
    print("OVERALL MODEL COMPARISON TABLE (MANDATORY SECTION 3.1)")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    # 4. Generate Visualizations (Mandatory Section 3.3)
    print("\n[4/4] Generating comparative visualization figures...")
    names = [r["model_name"] for r in all_results]
    cv_f1s = [r["cv_macro_f1"] for r in all_results]
    top1s = [r["student_test_top1_accuracy"] for r in all_results]
    test_f1s = [r["student_test_macro_f1"] for r in all_results]
    loglosses = [r["student_test_log_loss"] for r in all_results]
    continuities = [r["student_test_route_continuity"] for r in all_results]
    inf_times = [r["inference_time_s"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: CV Macro F1 vs Student-Test Macro F1
    x = np.arange(len(names))
    width = 0.35
    axes[0, 0].bar(x - width/2, cv_f1s, width, label="CV Macro F1", color="#1f77b4")
    axes[0, 0].bar(x + width/2, test_f1s, width, label="Student-Test Macro F1", color="#aec7e8")
    axes[0, 0].set_title("Cross-Validation vs Student-Test Macro F1 Comparison", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=25, ha="right")
    axes[0, 0].set_ylabel("Macro F1-score")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.7)
    for i in range(len(names)):
        axes[0, 0].text(x[i] - width/2, cv_f1s[i] + 0.02, f"{cv_f1s[i]:.3f}", ha="center", fontsize=9)
        axes[0, 0].text(x[i] + width/2, test_f1s[i] + 0.02, f"{test_f1s[i]:.3f}", ha="center", fontsize=9)

    # Plot 2: Observation-Level Top-1 Accuracy & Route Continuity
    axes[0, 1].bar(x - width/2, top1s, width, label="Top-1 Accuracy", color="#2ca02c")
    axes[0, 1].bar(x + width/2, continuities, width, label="Route Continuity", color="#d62728")
    axes[0, 1].set_title("Student-Test Top-1 Accuracy & Route Continuity", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=25, ha="right")
    axes[0, 1].set_ylabel("Fraction (0 to 1.0)")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.7)
    for i in range(len(names)):
        axes[0, 1].text(x[i] - width/2, top1s[i] + 0.02, f"{top1s[i]:.1%}", ha="center", fontsize=9)
        axes[0, 1].text(x[i] + width/2, continuities[i] + 0.02, f"{continuities[i]:.1%}", ha="center", fontsize=9)

    # Plot 3: Student-Test Log Loss Comparison
    axes[1, 0].bar(names, loglosses, color="#ff7f0e", width=0.5)
    axes[1, 0].set_title("Student-Test Log Loss (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticklabels(names, rotation=25, ha="right")
    axes[1, 0].set_ylabel("Log Loss")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)
    for i, v in enumerate(loglosses):
        axes[1, 0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    # Plot 4: Inference Speed & Training Time
    axes[1, 1].bar(names, [len(df_test)/t for t in inf_times], color="#9467bd", width=0.5)
    axes[1, 1].set_title("Inference Throughput (Observations per Second)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticklabels(names, rotation=25, ha="right")
    axes[1, 1].set_ylabel("Candidate Rows / Second")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)
    for i, t in enumerate(inf_times):
        rate = len(df_test) / t
        axes[1, 1].text(i, rate * 1.02, f"{rate:,.0f}/s", ha="center", fontsize=9)

    plt.tight_layout()
    comp_fig_path = MODELS_DIR / "model_comparison_charts.png"
    plt.savefig(comp_fig_path, dpi=150)
    plt.close()
    print(f"Comparison charts saved to: {comp_fig_path}")

    # Plot 5: All 5 Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (m_name, cm) in enumerate(confusion_matrices.items()):
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        im = axes[i].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        axes[i].set_title(f"{m_name}", fontsize=10, fontweight="bold")
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["P0", "P1"])
        axes[i].set_yticklabels(["T0", "T1"] if i == 0 else ["", ""])
        for r_idx in range(2):
            for c_idx in range(2):
                val_str = f"{cm_norm[r_idx, c_idx]:.1%}\n({cm[r_idx, c_idx]:,})"
                color = "white" if cm_norm[r_idx, c_idx] > 0.5 else "black"
                axes[i].text(c_idx, r_idx, val_str, ha="center", va="center", color=color, fontsize=8)
    plt.tight_layout()
    cm_fig_path = MODELS_DIR / "all_confusion_matrices.png"
    plt.savefig(cm_fig_path, dpi=150)
    plt.close()
    print(f"Confusion matrices saved to: {cm_fig_path}")

    print("\nALL MODELS BENCHMARK COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_benchmark()

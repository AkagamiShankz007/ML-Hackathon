import os
import time
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

def load_arcs_for_split(split_dir: Path) -> dict[str, np.ndarray]:
    arcs_dict = {}
    for trip_dir in split_dir.iterdir():
        if trip_dir.is_dir():
            trip_id = trip_dir.name
            arcs_path = trip_dir / f"{trip_id}.arcs"
            if arcs_path.exists():
                arcs_dict[trip_id] = np.loadtxt(arcs_path, dtype=np.int64, ndmin=2)
    return arcs_dict

def compute_top1_accuracy_and_continuity(
    df: pd.DataFrame, probs: np.ndarray, arcs_dict: dict[str, np.ndarray]
) -> tuple[float, float, dict]:
    temp_df = df[["trip_id", "observation_id", "candidate_arc", "reference_arc"]].copy()
    temp_df["prob"] = probs

    # Sort so top candidate per observation comes first
    idx = temp_df.groupby(["trip_id", "observation_id"])["prob"].idxmax()
    top1_df = temp_df.loc[idx].sort_values(["trip_id", "observation_id"])

    # Observation-level Top-1 accuracy
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

def train_and_evaluate():
    print("=" * 70)
    print("PATHSNAP: LOGISTIC REGRESSION TRAINING & BENCHMARK")
    print("=" * 70)

    # 1. Load Preprocessed Data
    parquet_path = ARTIFACTS / "candidate_features.parquet"
    print(f"\n[1/6] Loading dataset from {parquet_path}...")
    t_start = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} candidate rows in {time.time() - t_start:.2f}s")

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "student_test"

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    X_train = df_train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = df_train["label"].to_numpy(dtype=np.int8)
    groups_train = df_train["trip_id"].to_numpy()

    X_test = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = df_test["label"].to_numpy(dtype=np.int8)

    print(f"Train split: {len(df_train):,} rows across {df_train['trip_id'].nunique()} trips")
    print(f"Student-test split: {len(df_test):,} rows across {df_test['trip_id'].nunique()} trips")

    # 2. Group-Aware Cross-Validation (5 Folds on 80 Training Trips)
    print("\n[2/6] Running 5-Fold GroupKFold Cross-Validation on 80 training trips...")
    gkf = GroupKFold(n_splits=5)
    cv_macro_f1s = []
    cv_top1_accuracies = []
    cv_log_losses = []

    fold_idx = 1
    t_cv_start = time.time()
    for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
        t_fold = time.time()
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr_s, y_tr)

        val_probs = clf.predict_proba(X_val_s)[:, 1]
        val_preds = (val_probs >= 0.5).astype(np.int8)

        macro_f1 = f1_score(y_val, val_preds, average="macro")
        ll = log_loss(y_val, val_probs)

        # Fold top-1 accuracy
        val_sub_df = df_train.iloc[val_idx][["trip_id", "observation_id", "candidate_arc", "reference_arc"]].copy()
        val_sub_df["prob"] = val_probs
        top1_idx = val_sub_df.groupby(["trip_id", "observation_id"])["prob"].idxmax()
        fold_top1_acc = (val_sub_df.loc[top1_idx, "candidate_arc"] == val_sub_df.loc[top1_idx, "reference_arc"]).mean()

        cv_macro_f1s.append(macro_f1)
        cv_top1_accuracies.append(fold_top1_acc)
        cv_log_losses.append(ll)

        print(f"  Fold {fold_idx}: Macro F1 = {macro_f1:.4f}, Top-1 Acc = {fold_top1_acc:.4f}, LogLoss = {ll:.4f} ({time.time() - t_fold:.1f}s)")
        fold_idx += 1

    mean_cv_macro_f1 = float(np.mean(cv_macro_f1s))
    std_cv_macro_f1 = float(np.std(cv_macro_f1s))
    mean_cv_top1 = float(np.mean(cv_top1_accuracies))
    print(f"-> 5-Fold CV Macro F1: {mean_cv_macro_f1:.4f} +/- {std_cv_macro_f1:.4f} (Total CV Time: {time.time() - t_cv_start:.1f}s)")

    # 3. Fit Final Pipeline on Full 80 Training Trips
    print("\n[3/6] Fitting final model on full 80 training trips...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    t_fit_start = time.time()
    pipeline.fit(X_train, y_train)
    training_time_s = time.time() - t_fit_start
    print(f"Model fitting complete in {training_time_s:.2f}s")

    # 4. Evaluate on Student-Testing Partition (10 Trips)
    print("\n[4/6] Evaluating on 10 student-testing trips...")
    t_inf_start = time.time()
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    inference_time_s = time.time() - t_inf_start
    test_preds = (test_probs >= 0.5).astype(np.int8)

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    test_class1_f1 = f1_score(y_test, test_preds, pos_label=1)
    test_roc_auc = roc_auc_score(y_test, test_probs)
    test_logloss = log_loss(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)

    # 5. Observation-Level Top-1 Accuracy and Route Continuity
    print("\n[5/6] Computing Observation-level Top-1 Accuracy & Route Continuity...")
    student_test_arcs = load_arcs_for_split(STUDENT_TEST_DIR)
    top1_acc, route_continuity, trip_continuity = compute_top1_accuracy_and_continuity(
        df_test, test_probs, student_test_arcs
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION METRICS (STUDENT-TEST)")
    print("=" * 70)
    print(f"Algorithm Family:               Linear and Probabilistic")
    print(f"Model:                          Logistic Regression (StandardScaler + L2)")
    print(f"5-Fold CV Macro F1:             {mean_cv_macro_f1:.4f} (+/- {std_cv_macro_f1:.4f})")
    print(f"5-Fold CV Top-1 Accuracy:       {mean_cv_top1:.4f}")
    print(f"Student-Test Top-1 Accuracy:    {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"Student-Test Route Continuity:  {route_continuity:.4f} ({route_continuity*100:.2f}%)")
    print(f"Student-Test Macro F1-score:    {test_macro_f1:.4f}")
    print(f"Student-Test Positive F1-score: {test_class1_f1:.4f}")
    print(f"Student-Test Precision:         {test_prec:.4f}")
    print(f"Student-Test Recall:            {test_rec:.4f}")
    print(f"Student-Test Log Loss:          {test_logloss:.4f}")
    print(f"Student-Test ROC AUC:           {test_roc_auc:.4f}")
    print(f"Training Time:                  {training_time_s:.2f} seconds")
    print(f"Inference Time:                 {inference_time_s:.4f} seconds ({len(df_test)/inference_time_s:,.0f} rows/s)")
    print(f"Confusion Matrix (TN, FP, FN, TP):")
    print(cm)

    # Feature Importance (Coefficients)
    clf_model = pipeline.named_steps["classifier"]
    coefs = clf_model.coef_[0]
    print("\nLearned Feature Coefficients:")
    coef_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "coefficient": coefs})
    coef_df["abs_impact"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_impact", ascending=False)
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:<20}: {row['coefficient']:+.4f}")

    # 6. Save Model Artifacts, Metrics, and Evaluation Plots
    print("\n[6/6] Saving model artifacts, metrics, and figures...")
    model_path = MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    metrics_payload = {
        "model_name": "Logistic Regression",
        "algorithm_family": "Linear and probabilistic",
        "key_configuration": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 500,
            "scaler": "StandardScaler",
        },
        "cv_5fold_macro_f1_mean": mean_cv_macro_f1,
        "cv_5fold_macro_f1_std": std_cv_macro_f1,
        "cv_5fold_top1_acc_mean": mean_cv_top1,
        "student_test_top1_accuracy": top1_acc,
        "student_test_route_continuity": route_continuity,
        "student_test_macro_f1": test_macro_f1,
        "student_test_positive_f1": test_class1_f1,
        "student_test_precision": test_prec,
        "student_test_recall": test_rec,
        "student_test_log_loss": test_logloss,
        "student_test_roc_auc": test_roc_auc,
        "training_time_seconds": training_time_s,
        "inference_time_seconds": inference_time_s,
        "confusion_matrix": cm.tolist(),
        "coefficients": dict(zip(FEATURE_COLUMNS, coefs.tolist())),
        "per_trip_continuity": trip_continuity,
    }

    metrics_file = MODELS_DIR / "logistic_regression_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

    # Plot Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Coefficients plot
    axes[0].barh(coef_df["feature"], coef_df["coefficient"], color="#1f77b4")
    axes[0].axvline(0, color="gray", linestyle="--")
    axes[0].set_title("Logistic Regression Feature Weights (Coefficients)")
    axes[0].set_xlabel("Standardized Weight")

    # Confusion matrix normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    im = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Normalized Confusion Matrix (Student Test)")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Pred Neg", "Pred Pos"])
    axes[1].set_yticklabels(["True Neg", "True Pos"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2%}\n({cm[i, j]:,})", ha="center", va="center", color="red" if cm_norm[i, j] < 0.5 else "white")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    eval_plot_path = MODELS_DIR / "logistic_regression_eval.png"
    plt.savefig(eval_plot_path, dpi=150)
    plt.close()
    print(f"Diagnostic plot saved to: {eval_plot_path}")
    print("\nTraining and evaluation finished successfully!")

if __name__ == "__main__":
    train_and_evaluate()
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    log_loss, confusion_matrix
)

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

def load_arcs_cache():
    arcs_cache = {}
    data_root = "Student folder"
    for split in ["train", "student_test"]:
        split_dir = os.path.join(data_root, split)
        if os.path.exists(split_dir):
            for trip_name in os.listdir(split_dir):
                arcs_file = os.path.join(split_dir, trip_name, f"{trip_name}.arcs")
                if os.path.exists(arcs_file):
                    arcs_cache[trip_name] = np.loadtxt(arcs_file, dtype=int)
    return arcs_cache

def calculate_route_continuity(top1_df, arcs_cache):
    continuities = []
    for trip_id, group in top1_df.groupby("trip_id"):
        group_sorted = group.sort_values("observation_id")
        pred_arcs = group_sorted["candidate_arc"].values
        trip_key = str(trip_id).zfill(8)
        if trip_key not in arcs_cache:
            continue
        arcs = arcs_cache[trip_key]
        connected = 0
        total = len(pred_arcs) - 1
        if total <= 0:
            continuities.append(100.0)
            continue
        for i in range(total):
            a1, a2 = pred_arcs[i], pred_arcs[i+1]
            if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                connected += 1
        continuities.append((connected / total) * 100.0)
    return float(np.mean(continuities)) if continuities else 0.0

def train_improved_lightgbm():
    print("=" * 75)
    print("      OPTIMIZING LIGHTGBM WITH CLASS-WEIGHTED GRADIENT BOOSTING")
    print("=" * 75)

    # 1. Load Data
    train_csv = os.path.join("model_selection_data", "train.csv")
    test_csv = os.path.join("model_selection_data", "student_test.csv")
    parquet_path = os.path.join("artifacts", "candidate_features.parquet")
    cols = ["trip_id", "observation_id", "candidate_arc", "label"] + FEATURE_COLUMNS

    t0 = time.time()
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("\n[1/4] Loading preprocessed dataset from model_selection_data/...")
        df_train = pd.read_csv(train_csv, usecols=cols)
        df_test = pd.read_csv(test_csv, usecols=cols)
    else:
        print(f"\n[1/4] Loading preprocessed dataset from {parquet_path}...")
        df = pd.read_parquet(parquet_path, columns=["split"] + cols)
        df_train = df[df["split"] == "train"].copy()
        df_test = df[df["split"] == "student_test"].copy()

    df_train[FEATURE_COLUMNS] = df_train[FEATURE_COLUMNS].fillna(0)
    df_test[FEATURE_COLUMNS] = df_test[FEATURE_COLUMNS].fillna(0)

    # Add candidate relative ranking feature within observation
    df_train["obs_dist_rank"] = df_train.groupby(["trip_id", "observation_id"])["distance_m"].rank(method="min")
    df_test["obs_dist_rank"] = df_test.groupby(["trip_id", "observation_id"])["distance_m"].rank(method="min")
    
    # Add heading alignment score: cos(bearing_delta)
    df_train["heading_alignment"] = np.cos(np.radians(df_train["bearing_delta"]))
    df_test["heading_alignment"] = np.cos(np.radians(df_test["bearing_delta"]))

    ENHANCED_FEATURES = FEATURE_COLUMNS + ["obs_dist_rank", "heading_alignment"]

    print(f"  Training set:     {len(df_train):,} rows ({df_train['trip_id'].nunique()} trips)")
    print(f"  Student-test set: {len(df_test):,} rows ({df_test['trip_id'].nunique()} trips)")
    print(f"  Added engineered features: 'obs_dist_rank', 'heading_alignment'")

    # Sample 200,000 rows across trips for fast training
    np.random.seed(42)
    sample_size = min(200000, len(df_train))
    sample_idx = np.random.choice(len(df_train), size=sample_size, replace=False)
    df_train_sub = df_train.iloc[sample_idx]

    X_train_sub = df_train_sub[ENHANCED_FEATURES].values
    y_train_sub = df_train_sub["label"].values
    groups_train_sub = df_train_sub["trip_id"].values

    X_test = df_test[ENHANCED_FEATURES].values
    y_test = df_test["label"].values

    # 2. 5-Fold GroupKFold Cross-Validation
    print("\n[2/4] Running 5-Fold GroupKFold CV with optimized parameters...")
    gkf = GroupKFold(n_splits=5)
    cv_f1_scores = []
    fold = 0

    improved_params = {
        "n_estimators": 150,
        "learning_rate": 0.07,
        "num_leaves": 45,
        "max_depth": 8,
        "min_child_samples": 40,
        "scale_pos_weight": 3.0,  # Soft positive boost to tackle 1-vs-15 imbalance
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    for train_idx, val_idx in gkf.split(X_train_sub, y_train_sub, groups=groups_train_sub):
        fold += 1
        X_tr, y_tr = X_train_sub[train_idx], y_train_sub[train_idx]
        X_va, y_val = X_train_sub[val_idx], y_train_sub[val_idx]

        clf = lgb.LGBMClassifier(**improved_params)
        clf.fit(X_tr, y_tr)
        preds_va = clf.predict(X_va)
        f1 = f1_score(y_val, preds_va, average="macro")
        cv_f1_scores.append(f1)
        print(f"  Fold {fold}: Macro F1 = {f1:.4f}")

    mean_cv_f1 = float(np.mean(cv_f1_scores))
    print(f"  Mean CV Macro F1: {mean_cv_f1:.4f} (Baseline was 0.7196)")

    # 3. Train Full Model
    print("\n[3/4] Fitting full improved model...")
    t_start = time.time()
    model = lgb.LGBMClassifier(**improved_params)
    model.fit(X_train_sub, y_train_sub)
    train_time = time.time() - t_start
    print(f"  Trained in {train_time:.2f} seconds.")

    # 4. Evaluate on Student-Test
    print("\n[4/4] Evaluating on Student-Testing Partition (10 Trips)...")
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    test_precision = float(precision_score(y_test, test_preds, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds, zero_division=0))
    test_loss = float(log_loss(y_test, test_probs))

    # Observation-level Top-1 Accuracy
    df_test["predicted_score"] = test_probs
    idx_top1 = df_test.groupby(["trip_id", "observation_id"])["predicted_score"].idxmax()
    top1_df = df_test.loc[idx_top1]
    top1_accuracy = float((top1_df["label"] == 1).mean() * 100.0)

    arcs_cache = load_arcs_cache()
    route_continuity = calculate_route_continuity(top1_df, arcs_cache)

    print("\n" + "=" * 75)
    print("                IMPROVED LIGHTGBM vs BASELINE COMPARISON")
    print("=" * 75)
    print(f"  {'Metric':<32} | {'Baseline LightGBM':<18} | {'Improved LightGBM':<18}")
    print("  " + "-" * 71)
    print(f"  {'CV Macro F1-score':<32} | {'0.7196':<18} | {mean_cv_f1:<18.4f}")
    print(f"  {'Student-Test Top-1 Accuracy':<32} | {'57.88%':<18} | {top1_accuracy:<17.2f}%")
    print(f"  {'Route Continuity (Classifier)':<32} | {'79.87%':<18} | {route_continuity:<17.2f}%")
    print(f"  {'Student-Test Macro F1':<32} | {'0.7278':<18} | {test_macro_f1:<18.4f}")
    print(f"  {'Candidate Recall':<32} | {'42.82%':<18} | {test_recall * 100:<17.2f}%")
    print("=" * 75)

    # Save improved model
    joblib.dump(model, "artifacts/lightgbm_improved.joblib")
    print("\nSaved improved model to artifacts/lightgbm_improved.joblib")

if __name__ == "__main__":
    train_improved_lightgbm()

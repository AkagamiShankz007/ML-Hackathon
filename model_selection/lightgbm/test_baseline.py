import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, confusion_matrix, classification_report
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
    test_dir = os.path.join(data_root, "student_test")
    if os.path.exists(test_dir):
        for trip_name in os.listdir(test_dir):
            arcs_file = os.path.join(test_dir, trip_name, f"{trip_name}.arcs")
            if os.path.exists(arcs_file):
                arcs_cache[trip_name] = np.loadtxt(arcs_file, dtype=int)
    return arcs_cache

def calculate_trip_continuity(pred_arcs, arcs):
    connected = 0
    total = len(pred_arcs) - 1
    if total <= 0:
        return 100.0
    for i in range(total):
        a1, a2 = pred_arcs[i], pred_arcs[i+1]
        if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
            connected += 1
    return (connected / total) * 100.0

def test_lightgbm():
    print("=" * 80)
    print("        EVALUATING BASELINE LIGHTGBM ON STUDENT-TEST PARTITION")
    print("=" * 80)

    model_path = os.path.join("artifacts", "lightgbm.joblib")
    test_csv = os.path.join("model_selection_data", "student_test.csv")
    parquet_path = os.path.join("artifacts", "candidate_features.parquet")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        print("Please run 'python src/train_lightgbm.py' first.")
        return

    print("\n[Step 1] Loading baseline LightGBM model artifact...")
    model = joblib.load(model_path)
    print(f"  • Model loaded: {model_path}")

    print("\n[Step 2] Loading student-test dataset (10 trips)...")
    cols = ["trip_id", "observation_id", "candidate_arc", "label"] + FEATURE_COLUMNS
    if os.path.exists(test_csv):
        df_test = pd.read_csv(test_csv, usecols=cols)
    else:
        df = pd.read_parquet(parquet_path, columns=["split"] + cols)
        df_test = df[df["split"] == "student_test"].copy()

    df_test[FEATURE_COLUMNS] = df_test[FEATURE_COLUMNS].fillna(0)

    print(f"  • Test candidate rows: {len(df_test):,}")
    print(f"  • Test trips:          {df_test['trip_id'].nunique()}")
    print(f"  • Features:            {len(FEATURE_COLUMNS)}")

    print("\n[Step 3] Running inference...")
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test["label"].values

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    # Overall candidate-level metrics
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    loss = log_loss(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)

    # Observation-level Top-1 Accuracy and Route Continuity per trip
    print("\n[Step 4] Computing per-trip Top-1 Accuracy and Route Continuity...")
    df_test["predicted_score"] = test_probs
    idx_top1 = df_test.groupby(["trip_id", "observation_id"])["predicted_score"].idxmax()
    top1_df = df_test.loc[idx_top1]

    arcs_cache = load_arcs_cache()
    trip_metrics = []

    for trip_id, group in top1_df.groupby("trip_id"):
        group_sorted = group.sort_values("observation_id")
        n_obs = len(group_sorted)
        top1_acc = (group_sorted["label"] == 1).mean() * 100.0
        
        trip_key = str(trip_id).zfill(8)
        pred_arcs = group_sorted["candidate_arc"].values
        continuity = calculate_trip_continuity(pred_arcs, arcs_cache[trip_key]) if trip_key in arcs_cache else 0.0

        trip_metrics.append({
            "Trip ID": trip_key,
            "GPS Points": n_obs,
            "Top-1 Accuracy (%)": round(top1_acc, 2),
            "Route Continuity (%)": round(continuity, 2)
        })

    df_trip_metrics = pd.DataFrame(trip_metrics)

    print("\n" + "-" * 75)
    print(f"{'Trip ID':<12} | {'GPS Points':<12} | {'Top-1 Accuracy':<18} | {'Route Continuity'}")
    print("-" * 75)
    for _, row in df_trip_metrics.iterrows():
        print(f"{row['Trip ID']:<12} | {row['GPS Points']:<12} | {row['Top-1 Accuracy (%)']:<6.2f}%            | {row['Route Continuity (%)']:<6.2f}%")
    print("-" * 75)

    overall_top1 = (top1_df["label"] == 1).mean() * 100.0
    overall_continuity = df_trip_metrics["Route Continuity (%)"].mean()

    # Feature Importance
    print("\n[Learned Feature Importance (Split Count)]")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        feat_name = FEATURE_COLUMNS[idx]
        print(f"  • {feat_name:<20}: {importances[idx]:>5} splits")

    # Summary
    print("\n" + "=" * 80)
    print("                     BASELINE LIGHTGBM: TEST RESULTS")
    print("=" * 80)
    print(f"  • Overall Observation Top-1 Accuracy:  {overall_top1:.2f}%")
    print(f"  • Overall Route Continuity:            {overall_continuity:.2f}%")
    print(f"  • Candidate-Level Macro F1-score:      {macro_f1:.4f}")
    print(f"  • Precision (Traversed Class):         {precision:.4f}")
    print(f"  • Recall (Traversed Class):            {recall * 100:.2f}%")
    print(f"  • Log Loss (Cross-Entropy):            {loss:.4f}")
    print(f"  • Total Test Candidates Evaluated:     {len(y_test):,}")
    print("-" * 80)
    print("\n[Candidate Classification Report]")
    print(classification_report(y_test, test_preds, target_names=["Non-traversed (0)", "Traversed (1)"]))

    print("[Confusion Matrix]")
    print(f"  TN (Correctly rejected false arcs): {cm[0, 0]:,}")
    print(f"  FP (False alarm road candidates):   {cm[0, 1]:,}")
    print(f"  FN (Missed ground-truth arcs):      {cm[1, 0]:,}")
    print(f"  TP (Correctly identified arcs):     {cm[1, 1]:,}")
    print("=" * 80)

if __name__ == "__main__":
    test_lightgbm()

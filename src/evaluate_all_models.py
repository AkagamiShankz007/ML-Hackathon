import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
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
    test_dir = os.path.join(data_root, "student_test")
    if os.path.exists(test_dir):
        for trip_name in os.listdir(test_dir):
            arcs_file = os.path.join(test_dir, trip_name, f"{trip_name}.arcs")
            if os.path.exists(arcs_file):
                arcs_cache[trip_name] = np.loadtxt(arcs_file, dtype=int)
    return arcs_cache

def calculate_continuity(top1_df, arcs_cache):
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

def evaluate_all():
    print("=" * 105)
    print("      CONSOLIDATED MODEL EVALUATION (PES HACKATHON SECTION 3.1 BENCHMARK)")
    print("=" * 105)

    test_csv = os.path.join("model_selection_data", "student_test.csv")
    parquet_path = os.path.join("artifacts", "candidate_features.parquet")
    cols = ["trip_id", "observation_id", "candidate_arc", "label"] + FEATURE_COLUMNS

    if os.path.exists(test_csv):
        df_test = pd.read_csv(test_csv, usecols=cols)
    else:
        df = pd.read_parquet(parquet_path, columns=["split"] + cols)
        df_test = df[df["split"] == "student_test"].copy()

    df_test[FEATURE_COLUMNS] = df_test[FEATURE_COLUMNS].fillna(0)
    df_test["obs_dist_rank"] = df_test.groupby(["trip_id", "observation_id"])["distance_m"].rank(method="min")
    df_test["heading_alignment"] = np.cos(np.radians(df_test["bearing_delta"]))
    df_test["speed_aligned"] = df_test["gps_speed_mps"] * df_test["heading_alignment"]

    y_test = df_test["label"].values
    arcs_cache = load_arcs_cache()

    models_info = [
        {
            "name": "Logistic Reg (Baseline)",
            "family": "Linear / Probabilistic",
            "config": "StandardScaler, C=1.0, L2",
            "cv_f1": 0.6443,
            "model_path": "artifacts/logistic_regression.joblib",
            "scaler_path": "artifacts/scaler_logistic.joblib",
            "features": FEATURE_COLUMNS,
            "needs_scale": True
        },
        {
            "name": "Logistic Reg (Improved)",
            "family": "Linear / Probabilistic",
            "config": "Balanced weights, rank+cos features",
            "cv_f1": 0.6828,
            "model_path": "artifacts/logistic_regression_improved.joblib",
            "scaler_path": "artifacts/scaler_logistic_improved.joblib",
            "features": FEATURE_COLUMNS + ["obs_dist_rank", "heading_alignment", "speed_aligned"],
            "needs_scale": True
        },
        {
            "name": "LightGBM (Baseline)",
            "family": "Gradient Boosted Trees",
            "config": "100 trees, lr=0.1, max_depth=6",
            "cv_f1": 0.7196,
            "model_path": "artifacts/lightgbm.joblib",
            "scaler_path": None,
            "features": FEATURE_COLUMNS,
            "needs_scale": False
        },
        {
            "name": "LightGBM (Improved)",
            "family": "Gradient Boosted Trees",
            "config": "150 trees, pos_weight=3.0, rank+cos",
            "cv_f1": 0.8117,
            "model_path": "artifacts/lightgbm_improved.joblib",
            "scaler_path": None,
            "features": FEATURE_COLUMNS + ["obs_dist_rank", "heading_alignment"],
            "needs_scale": False
        },
        {
            "name": "Bagging (Trees)",
            "family": "Bootstrap Aggregation",
            "config": "20 trees, depth 14, balanced",
            "cv_f1": 0.7725,
            "model_path": "artifacts/model_selection/bagging/bagging_model.joblib",
            "scaler_path": None,
            "features": FEATURE_COLUMNS,
            "needs_scale": False
        }
    ]

    results = []

    for m in models_info:
        if not os.path.exists(m["model_path"]):
            continue
        model = joblib.load(m["model_path"])
        X = df_test[m["features"]].values

        if m["needs_scale"] and os.path.exists(m["scaler_path"]):
            scaler = joblib.load(m["scaler_path"])
            t0 = time.time()
            X_eval = scaler.transform(X)
            probs = model.predict_proba(X_eval)[:, 1]
            preds = model.predict(X_eval)
            inf_time_total = time.time() - t0
        else:
            t0 = time.time()
            probs = model.predict_proba(X)[:, 1]
            preds = model.predict(X)
            inf_time_total = time.time() - t0

        ms_per_1k = (inf_time_total / len(y_test)) * 1000 * 1000

        macro_f1 = f1_score(y_test, preds, average="macro")
        loss = log_loss(y_test, probs)

        df_test["predicted_score"] = probs
        idx_top1 = df_test.groupby(["trip_id", "observation_id"])["predicted_score"].idxmax()
        top1_df = df_test.loc[idx_top1]
        top1_acc = (top1_df["label"] == 1).mean() * 100.0
        continuity = calculate_continuity(top1_df, arcs_cache)

        results.append({
            "Model": m["name"],
            "Algorithm Family": m["family"],
            "Key Configuration": m["config"],
            "CV Macro F1": f"{m['cv_f1']:.4f}",
            "Top-1 Acc": f"{top1_acc:.2f}%",
            "Route Cont": f"{continuity:.2f}%",
            "Test Macro F1": f"{macro_f1:.4f}",
            "Log Loss": f"{loss:.4f}",
            "Inference": f"{ms_per_1k:.2f} ms/1k"
        })

    df_res = pd.DataFrame(results)
    print("\n" + "-" * 115)
    print(f"{'Model':<25} | {'Algorithm Family':<22} | {'CV F1':<7} | {'Top-1':<7} | {'Cont.':<7} | {'Test F1':<7} | {'LogLoss':<7} | {'Inference':<10}")
    print("-" * 115)
    for _, r in df_res.iterrows():
        print(f"{r['Model']:<25} | {r['Algorithm Family']:<22} | {r['CV Macro F1']:<7} | {r['Top-1 Acc']:<7} | {r['Route Cont']:<7} | {r['Test Macro F1']:<7} | {r['Log Loss']:<7} | {r['Inference']:<10}")
    print("-" * 115)

    out_csv = "artifacts/consolidated_model_comparison.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved consolidated table to {out_csv}")

if __name__ == "__main__":
    evaluate_all()

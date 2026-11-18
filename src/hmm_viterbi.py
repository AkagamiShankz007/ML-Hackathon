import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
STUDENT_TEST_DIR = ROOT / "Student folder" / "student_test"
EARTH_RADIUS_M = 6_371_000.0

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


def local_xy(lon, lat, lon0, lat0):
    lat0_rad = np.radians(lat0)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(lat0_rad)
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


class RoadNetworkHMM:
    """Hidden Markov Model for Sequential Map-Matching & Route Reconstruction."""

    def __init__(self, beta: float = 30.0, p_stay: float = 0.65, p_connected: float = 0.35):
        self.beta = beta
        self.p_stay = p_stay
        self.p_connected = p_connected

    def decode_trip(
        self,
        observations_df: pd.DataFrame,
        classifier_probs: np.ndarray,
        arcs: np.ndarray,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        temp_df = observations_df.copy()
        temp_df["prob"] = classifier_probs

        lon0 = float(np.mean(nodes[:, 0]))
        lat0 = float(np.mean(nodes[:, 1]))
        nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)

        from_nodes = arcs[:, 0]
        to_nodes = arcs[:, 1]
        ax, ay = nx[from_nodes], ny[from_nodes]
        bx, by = nx[to_nodes], ny[to_nodes]

        grouped = list(temp_df.groupby("observation_id"))
        T = len(grouped)
        if T == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        classifier_top1 = np.zeros(T, dtype=np.int64)
        viterbi_path = np.zeros(T, dtype=np.int64)

        step_candidates = []
        step_emissions = []
        step_gaps = []

        for t, (obs_id, obs_rows) in enumerate(grouped):
            cand_arcs = obs_rows["candidate_arc"].to_numpy(dtype=np.int64)
            c_probs = obs_rows["prob"].to_numpy(dtype=np.float64)

            best_idx = np.argmax(c_probs)
            classifier_top1[t] = cand_arcs[best_idx]

            p_sum = c_probs.sum()
            if p_sum > 0:
                norm_p = c_probs / p_sum
            else:
                norm_p = np.full(len(c_probs), 1.0 / len(c_probs))

            log_emiss = np.log(np.clip(norm_p, 1e-12, 1.0))
            gap = float(obs_rows["log_gps_gap_s"].iloc[0])

            step_candidates.append(cand_arcs)
            step_emissions.append(log_emiss)
            step_gaps.append(gap)

        V = [step_emissions[0]]
        backpointers = []
        log_p_stay = np.log(self.p_stay)
        log_p_conn = np.log(self.p_connected)

        for t in range(1, T):
            prev_cands = step_candidates[t - 1]
            curr_cands = step_candidates[t]
            curr_emiss = step_emissions[t]
            gap_val = step_gaps[t]

            K_prev = len(prev_cands)
            K_curr = len(curr_cands)

            is_large_gap = gap_val > np.log1p(5.0)
            log_trans = np.zeros((K_prev, K_curr), dtype=np.float64)

            if is_large_gap:
                log_trans.fill(0.0)
            else:
                for i in range(K_prev):
                    a_id = prev_cands[i]
                    a_to = to_nodes[a_id]
                    bx_a = bx[a_id]
                    by_a = by[a_id]

                    for j in range(K_curr):
                        b_id = curr_cands[j]
                        b_from = from_nodes[b_id]

                        if a_id == b_id:
                            log_trans[i, j] = log_p_stay
                        elif a_to == b_from:
                            log_trans[i, j] = log_p_conn
                        else:
                            ax_b = ax[b_id]
                            ay_b = ay[b_id]
                            d = np.sqrt((bx_a - ax_b) ** 2 + (by_a - ay_b) ** 2)
                            log_trans[i, j] = log_p_conn - (d / self.beta) - 4.0

            prev_V = V[t - 1][:, np.newaxis]
            total_prob = prev_V + log_trans
            bp = np.argmax(total_prob, axis=0)
            max_val = np.max(total_prob, axis=0) + curr_emiss

            V.append(max_val)
            backpointers.append(bp)

        best_last = np.argmax(V[-1])
        viterbi_path[-1] = step_candidates[-1][best_last]
        curr_ptr = best_last

        for t in range(T - 2, -1, -1):
            curr_ptr = backpointers[t][curr_ptr]
            viterbi_path[t] = step_candidates[t][curr_ptr]

        return classifier_top1, viterbi_path


def evaluate_continuity_metrics(predicted_arcs: np.ndarray, arcs: np.ndarray) -> dict:
    if len(predicted_arcs) <= 1:
        return {"obs_continuity": 1.0, "dedup_continuity": 1.0, "raw_connected": 0, "total_trans": 0}

    valid = (predicted_arcs >= 0) & (predicted_arcs < len(arcs))
    valid_pairs = valid[:-1] & valid[1:]
    pair_idx = np.flatnonzero(valid_pairs)
    n_trans = len(predicted_arcs) - 1

    # 1. Connected transitions (to == from)
    connected = int(np.sum(arcs[predicted_arcs[pair_idx], 1] == arcs[predicted_arcs[pair_idx + 1], 0]))

    # 2. Observation continuity (stay on same arc OR connected to next)
    obs_continuous = int(np.sum(
        (predicted_arcs[pair_idx] == predicted_arcs[pair_idx + 1]) |
        (arcs[predicted_arcs[pair_idx], 1] == arcs[predicted_arcs[pair_idx + 1], 0])
    ))

    # 3. Deduplicated route transitions
    dedup = predicted_arcs[np.concatenate(([True], predicted_arcs[1:] != predicted_arcs[:-1]))]
    dedup_trans = max(1, len(dedup) - 1)
    dedup_connected = int(np.sum(arcs[dedup[:-1], 1] == arcs[dedup[1:], 0]))

    return {
        "raw_connected": connected,
        "total_trans": n_trans,
        "raw_connected_rate": float(connected / n_trans) if n_trans else 1.0,
        "obs_continuity": float(obs_continuous / n_trans) if n_trans else 1.0,
        "dedup_continuity": float(dedup_connected / dedup_trans),
        "unique_arcs_traversed": int(len(dedup)),
    }


def run_hmm_benchmark():
    print("=" * 80)
    print("PATHSNAP: HMM & VITERBI SEQUENTIAL ROUTE RECONSTRUCTION")
    print("=" * 80)

    model_path = MODELS_DIR / "lightgbm_tuned.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "lightgbm.joblib"
    print(f"\n[1/4] Loading classification model from {model_path.name}...")
    model = joblib.load(model_path)

    parquet_path = ARTIFACTS / "candidate_features.parquet"
    print(f"Loading student-test candidate features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df_test = df[df["split"] == "student_test"].reset_index(drop=True)

    X_test = np.ascontiguousarray(df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    t0 = time.time()
    test_probs = model.predict_proba(X_test)[:, 1]
    df_test["model_prob"] = test_probs
    print(f"Computed candidate probabilities for {len(df_test):,} candidate rows in {time.time() - t0:.2f}s")

    print("\n[2/4] Decoding routes across all 10 student-test trips with Viterbi...")
    hmm = RoadNetworkHMM(beta=30.0, p_stay=0.65, p_connected=0.35)

    trip_results = []
    total_obs = 0
    clf_correct_total = 0
    hmm_correct_total = 0

    t_decode_start = time.time()
    for trip_dir in sorted(STUDENT_TEST_DIR.iterdir()):
        if not trip_dir.is_dir():
            continue
        trip_id = trip_dir.name
        trip_df = df_test[df_test["trip_id"] == trip_id].copy()
        if len(trip_df) == 0:
            continue

        arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
        nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64, ndmin=2)

        clf_preds, hmm_preds = hmm.decode_trip(
            trip_df, trip_df["model_prob"].to_numpy(), arcs, nodes
        )

        ref_arcs = trip_df.groupby("observation_id")["reference_arc"].first().to_numpy(dtype=np.int64)
        n_obs = len(ref_arcs)
        total_obs += n_obs

        clf_acc = float((clf_preds == ref_arcs).mean())
        hmm_acc = float((hmm_preds == ref_arcs).mean())
        clf_correct_total += int((clf_preds == ref_arcs).sum())
        hmm_correct_total += int((hmm_preds == ref_arcs).sum())

        clf_metrics = evaluate_continuity_metrics(clf_preds, arcs)
        hmm_metrics = evaluate_continuity_metrics(hmm_preds, arcs)

        trip_results.append({
            "trip_id": trip_id,
            "observations": n_obs,
            "clf_top1_acc": clf_acc,
            "hmm_top1_acc": hmm_acc,
            "delta_acc": hmm_acc - clf_acc,
            "clf_obs_continuity": clf_metrics["obs_continuity"],
            "hmm_obs_continuity": hmm_metrics["obs_continuity"],
            "clf_dedup_continuity": clf_metrics["dedup_continuity"],
            "hmm_dedup_continuity": hmm_metrics["dedup_continuity"],
            "clf_raw_connected_rate": clf_metrics["raw_connected_rate"],
            "hmm_raw_connected_rate": hmm_metrics["raw_connected_rate"],
        })

        print(f"  Trip {trip_id} ({n_obs} obs): Acc: {clf_acc:.1%} -> {hmm_acc:.1%} | Continuity: {clf_metrics['obs_continuity']:.1%} -> {hmm_metrics['obs_continuity']:.1%}")

    decode_time = time.time() - t_decode_start
    overall_clf_acc = clf_correct_total / total_obs
    overall_hmm_acc = hmm_correct_total / total_obs
    overall_clf_obs_cont = float(np.mean([r["clf_obs_continuity"] for r in trip_results]))
    overall_hmm_obs_cont = float(np.mean([r["hmm_obs_continuity"] for r in trip_results]))
    overall_clf_dedup_cont = float(np.mean([r["clf_dedup_continuity"] for r in trip_results]))
    overall_hmm_dedup_cont = float(np.mean([r["hmm_dedup_continuity"] for r in trip_results]))

    print("\n" + "=" * 80)
    print("CLASSIFIER-ONLY VS CLASSIFIER + HMM/VITERBI COMPARISON (GUIDELINES § 5)")
    print("=" * 80)
    summary_comparison = pd.DataFrame([
        {
            "Metric": "Observation-Level Top-1 Accuracy",
            "Classifier Only": f"{overall_clf_acc:.4f} ({overall_clf_acc*100:.2f}%)",
            "Classifier + HMM/Viterbi": f"{overall_hmm_acc:.4f} ({overall_hmm_acc*100:.2f}%)",
            "Absolute Improvement": f"{overall_hmm_acc - overall_clf_acc:+.4f} ({(overall_hmm_acc - overall_clf_acc)*100:+.2f}%)",
        },
        {
            "Metric": "Observation Trajectory Continuity",
            "Classifier Only": f"{overall_clf_obs_cont:.4f} ({overall_clf_obs_cont*100:.2f}%)",
            "Classifier + HMM/Viterbi": f"{overall_hmm_obs_cont:.4f} ({overall_hmm_obs_cont*100:.2f}%)",
            "Absolute Improvement": f"{overall_hmm_obs_cont - overall_clf_obs_cont:+.4f} ({(overall_hmm_obs_cont - overall_clf_obs_cont)*100:+.2f}%)",
        },
        {
            "Metric": "Deduplicated Route Topological Continuity",
            "Classifier Only": f"{overall_clf_dedup_cont:.4f} ({overall_clf_dedup_cont*100:.2f}%)",
            "Classifier + HMM/Viterbi": f"{overall_hmm_dedup_cont:.4f} ({overall_hmm_dedup_cont*100:.2f}%)",
            "Absolute Improvement": f"{overall_hmm_dedup_cont - overall_clf_dedup_cont:+.4f} ({(overall_hmm_dedup_cont - overall_clf_dedup_cont)*100:+.2f}%)",
        },
    ])
    print(summary_comparison.to_string(index=False))

    per_trip_df = pd.DataFrame(trip_results)
    per_trip_df.to_csv(MODELS_DIR / "hmm_per_trip_comparison.csv", index=False)
    summary_comparison.to_csv(MODELS_DIR / "hmm_overall_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(trip_results))
    width = 0.35
    trip_names = [r["trip_id"][-4:] for r in trip_results]

    axes[0].bar(x - width/2, [r["clf_top1_acc"] for r in trip_results], width, label="Classifier Only", color="#aec7e8")
    axes[0].bar(x + width/2, [r["hmm_top1_acc"] for r in trip_results], width, label="Classifier + HMM", color="#1f77b4")
    axes[0].set_title("Pointwise Top-1 Accuracy: Clf vs HMM (Per Trip)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trip_names, rotation=45)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    axes[1].bar(x - width/2, [r["clf_obs_continuity"] for r in trip_results], width, label="Classifier Only", color="#ffbb78")
    axes[1].bar(x + width/2, [r["hmm_obs_continuity"] for r in trip_results], width, label="Classifier + HMM", color="#2ca02c")
    axes[1].set_title("Trajectory Continuity: Clf vs HMM (Per Trip)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(trip_names, rotation=45)
    axes[1].set_ylabel("Continuity Fraction")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    hmm_plot_path = MODELS_DIR / "hmm_route_reconstruction_eval.png"
    plt.savefig(hmm_plot_path, dpi=150)
    plt.close()
    print(f"HMM Comparison plot saved to: {hmm_plot_path}")

    hmm_payload = {
        "overall_classifier_top1_accuracy": overall_clf_acc,
        "overall_hmm_top1_accuracy": overall_hmm_acc,
        "overall_classifier_obs_continuity": overall_clf_obs_cont,
        "overall_hmm_obs_continuity": overall_hmm_obs_cont,
        "overall_classifier_dedup_continuity": overall_clf_dedup_cont,
        "overall_hmm_dedup_continuity": overall_hmm_dedup_cont,
        "decoding_time_seconds": decode_time,
        "observations_processed": total_obs,
        "per_trip_results": trip_results,
    }
    with open(MODELS_DIR / "hmm_reconstruction_metrics.json", "w") as f:
        json.dump(hmm_payload, f, indent=2)

    print("\nHMM Route Reconstruction finished successfully!")


if __name__ == "__main__":
    run_hmm_benchmark()

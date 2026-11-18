from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
import joblib
from branca.element import Element
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from build_features import FEATURE_COLUMNS, clean_track, load_trip


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "model_selection_data"
OUTPUT_DIR = ROOT / "artifacts" / "model_selection" / "bagging"
RANDOM_STATE = 42
CV_SPLITS = 5
N_ESTIMATORS = 20
MAX_SAMPLES = 0.10
MAX_FEATURES = 1.0
N_JOBS = 4
ID_COLUMNS = ("trip_id", "observation_id", "candidate_arc")
MODEL_COLUMNS = [*ID_COLUMNS, *FEATURE_COLUMNS, "label"]


def load_partition(split: str):
    dtype = {name: np.float32 for name in FEATURE_COLUMNS}
    dtype.update({
        "trip_id": str,
        "observation_id": np.int32,
        "candidate_arc": np.int32,
        "label": np.int8,
    })
    frame = pd.read_csv(
        DATA_DIR / f"{split}.csv",
        usecols=MODEL_COLUMNS,
        dtype=dtype,
    )
    features = np.ascontiguousarray(
        frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    )
    labels = frame["label"].to_numpy(dtype=np.int8)
    return frame, features, labels


def make_model(random_state: int) -> BaggingClassifier:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=14,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
    )
    return BaggingClassifier(
        estimator=tree,
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=N_JOBS,
        random_state=random_state,
    )


def positive_probability(model: BaggingClassifier, features: np.ndarray):
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(features)[:, positive_index]


def cross_validate(features, labels, groups):
    splitter = StratifiedGroupKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_rows = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels, groups), start=1
    ):
        train_groups = np.unique(groups[train_indices])
        validation_groups = np.unique(groups[validation_indices])
        if np.intersect1d(train_groups, validation_groups).size:
            raise RuntimeError(f"Trip overlap found in CV fold {fold_number}")

        model = make_model(RANDOM_STATE + fold_number)
        fit_start = time.perf_counter()
        model.fit(features[train_indices], labels[train_indices])
        fit_seconds = time.perf_counter() - fit_start

        train_predictions = model.predict(features[train_indices])
        validation_predictions = model.predict(features[validation_indices])
        validation_probabilities = positive_probability(
            model, features[validation_indices]
        )
        fold_rows.append({
            "fold": fold_number,
            "training_trips": int(train_groups.size),
            "validation_trips": int(validation_groups.size),
            "training_candidates": int(train_indices.size),
            "validation_candidates": int(validation_indices.size),
            "training_macro_f1": float(f1_score(
                labels[train_indices],
                train_predictions,
                average="macro",
                zero_division=0,
            )),
            "validation_macro_f1": float(f1_score(
                labels[validation_indices],
                validation_predictions,
                average="macro",
                zero_division=0,
            )),
            "validation_log_loss": float(log_loss(
                labels[validation_indices],
                validation_probabilities,
                labels=[0, 1],
            )),
            "fit_seconds": float(fit_seconds),
        })
        print(
            f"CV fold {fold_number}/{CV_SPLITS}: "
            f"Macro F1={fold_rows[-1]['validation_macro_f1']:.4f}; "
            f"fit={fit_seconds:.1f}s; "
            f"trips={train_groups.size}/{validation_groups.size}"
        )
        del model, train_predictions, validation_predictions
        del validation_probabilities
        gc.collect()
    return pd.DataFrame(fold_rows)


def top1_predictions(test_frame: pd.DataFrame, scores: np.ndarray):
    candidates = test_frame.loc[:, [*ID_COLUMNS, "label"]].copy()
    candidates["positive_probability"] = scores
    candidates = candidates.sort_values(
        ["trip_id", "observation_id", "candidate_arc"],
        kind="mergesort",
    )
    selected_indices = candidates.groupby(
        ["trip_id", "observation_id"], sort=False
    )["positive_probability"].idxmax()
    selected = candidates.loc[selected_indices].copy()
    selected["correct"] = selected["label"].eq(1)
    return selected.sort_values(
        ["trip_id", "observation_id"], kind="mergesort"
    ).reset_index(drop=True), candidates


def measure_route_continuity(top1: pd.DataFrame):
    connected_count = 0
    eligible_count = 0
    skipped_long_gap_count = 0
    for trip_id, trip_predictions in top1.groupby("trip_id", sort=True):
        track, _, arcs, _ = load_trip("student_test", trip_id)
        track, _ = clean_track(track, trip_id)
        ordered = trip_predictions.sort_values("observation_id")
        observation_ids = ordered["observation_id"].to_numpy(dtype=np.int64)
        predicted_arcs = ordered["candidate_arc"].to_numpy(dtype=np.int64)
        if len(observation_ids) != len(track) or not np.array_equal(
            observation_ids, np.arange(len(track))
        ):
            raise ValueError(f"Prediction observations do not align for trip {trip_id}")
        if np.any((predicted_arcs < 0) | (predicted_arcs >= len(arcs))):
            raise ValueError(f"Invalid predicted arc ID for trip {trip_id}")

        gaps = np.diff(track[:, 2])
        short_gap = gaps <= 5.0
        skipped_long_gap_count += int((~short_gap).sum())
        connected = (
            (predicted_arcs[:-1] == predicted_arcs[1:])
            | (arcs[predicted_arcs[:-1], 1] == arcs[predicted_arcs[1:], 0])
        )
        eligible_count += int(short_gap.sum())
        connected_count += int((connected & short_gap).sum())

    return {
        "connected_transitions": connected_count,
        "eligible_transitions": eligible_count,
        "continuity": (
            float(connected_count / eligible_count) if eligible_count else 0.0
        ),
        "long_gap_transitions_excluded": skipped_long_gap_count,
        "definition": (
            "Fraction of adjacent observations with <=5 second gaps whose "
            "predicted arcs are identical or directly connected in the directed graph."
        ),
    }


def add_arc_layer(group, nodes, arcs, arc_ids, color, weight, opacity, name):
    for arc_id in arc_ids:
        if 0 <= int(arc_id) < len(arcs):
            endpoints = arcs[int(arc_id)]
            coordinates = [
                [float(nodes[endpoints[0], 1]), float(nodes[endpoints[0], 0])],
                [float(nodes[endpoints[1], 1]), float(nodes[endpoints[1], 0])],
            ]
            folium.PolyLine(
                coordinates,
                color=color,
                weight=weight,
                opacity=opacity,
                tooltip=f"{name} arc {int(arc_id)}",
            ).add_to(group)


def create_prediction_map(trip_id: str, top1: pd.DataFrame):
    track, nodes, arcs, reference_route = load_trip("student_test", trip_id)
    track, _ = clean_track(track, trip_id)
    map_obj = folium.Map(
        location=[float(track[:, 1].mean()), float(track[:, 0].mean())],
        zoom_start=14,
        control_scale=True,
    )

    network_layer = folium.FeatureGroup(name="Road network", show=True)
    add_arc_layer(
        network_layer, nodes, arcs, range(len(arcs)),
        color="#777777", weight=1, opacity=0.35, name="Road network",
    )

    reference_layer = folium.FeatureGroup(name="Supplied reference route", show=True)
    if reference_route is not None:
        add_arc_layer(
            reference_layer, nodes, arcs, reference_route,
            color="#2ca02c", weight=3, opacity=0.85, name="Reference",
        )

    predicted_layer = folium.FeatureGroup(name="Bagging top-1 predictions", show=True)
    selected = top1.loc[top1["trip_id"] == trip_id].sort_values("observation_id")
    add_arc_layer(
        predicted_layer, nodes, arcs, selected["candidate_arc"].to_numpy(),
        color="#d62728", weight=4, opacity=0.95, name="Bagging prediction",
    )

    gps_layer = folium.FeatureGroup(name="GPS trajectory", show=True)
    break_indices = np.flatnonzero(np.diff(track[:, 2]) > 5.0) + 1
    for segment in np.split(track, break_indices):
        if len(segment) > 1:
            folium.PolyLine(
                segment[:, [1, 0]].tolist(),
                color="#1f77b4",
                weight=1.5,
                opacity=0.55,
                tooltip="GPS trajectory (gaps >5 s broken)",
            ).add_to(gps_layer)
    marker_step = max(1, len(track) // 60)
    for longitude, latitude, _ in track[::marker_step]:
        folium.CircleMarker(
            location=[float(latitude), float(longitude)],
            radius=1.5,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.7,
        ).add_to(gps_layer)

    gps_layer.add_to(map_obj)
    network_layer.add_to(map_obj)
    reference_layer.add_to(map_obj)
    predicted_layer.add_to(map_obj)

    map_obj.fit_bounds([
        [float(track[:, 1].min()), float(track[:, 0].min())],
        [float(track[:, 1].max()), float(track[:, 0].max())],
    ])
    map_obj.get_root().html.add_child(Element(
        f"<h4 style='position:fixed;top:10px;left:60px;z-index:9999;"
        f"background:white;padding:6px;border:1px solid #777'>"
        f"Bagging map-matching | student_test trip {trip_id}</h4>"
    ))
    legend = """
    <div style="position:fixed;bottom:25px;left:25px;z-index:9999;
                background:white;padding:9px 12px;border:1px solid #777;
                border-radius:4px;font-size:13px">
      <strong>PathSnap layers</strong><br>
      <span style="color:#777">━━</span> Road network<br>
      <span style="color:#1f77b4">━━</span> GPS trajectory / fixes<br>
      <span style="color:#2ca02c">━━</span> Supplied reference route<br>
      <span style="color:#d62728">━━</span> Bagging top-1 prediction
    </div>
    """
    map_obj.get_root().html.add_child(Element(legend))
    folium.LayerControl(collapsed=False).add_to(map_obj)
    output = OUTPUT_DIR / f"bagging_student_test_trip_{trip_id}.html"
    map_obj.save(output)
    return output


def write_plots(fold_frame, metrics, confusion, model):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fold_scores = fold_frame["validation_macro_f1"].to_numpy()
    mean_score = float(fold_scores.mean())
    std_score = float(fold_scores.std(ddof=1))
    ax.bar(
        ["Bagging"],
        [mean_score],
        yerr=[std_score],
        capsize=8,
        color="#4C78A8",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Candidate-level Macro F1")
    ax.set_title("Bagging: 5-fold trip-grouped CV Macro F1")
    ax.text(0, min(mean_score + 0.03, 0.97), f"{mean_score:.3f} ± {std_score:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cv_macro_f1.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Student-test\nobservation Top-1", "Student-test\ncandidate Macro F1"]
    values = [
        metrics["student_test_top1_accuracy"],
        metrics["student_test_candidate_macro_f1"],
    ]
    ax.bar(labels, values, color=["#59A14F", "#F28E2B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Bagging student-test metrics (different evaluation units)")
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "student_test_metrics.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(
        ["Bagging"],
        [metrics["student_test_candidate_log_loss"]],
        color="#E15759",
    )
    ax.set_ylabel("Candidate-level Log Loss (lower is better)")
    ax.set_title("Bagging student-test Log Loss")
    ax.text(
        0,
        metrics["student_test_candidate_log_loss"],
        f" {metrics['student_test_candidate_log_loss']:.3f}",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "student_test_log_loss.png", dpi=170)
    plt.close(fig)

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Non-reference", "Reference"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    display.ax_.set_title("Bagging student-test candidate confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(OUTPUT_DIR / "student_test_confusion_matrix.png", dpi=170)
    plt.close(display.figure_)

    importances = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    for estimator, feature_indices in zip(
        model.estimators_, model.estimators_features_
    ):
        importances[np.asarray(feature_indices, dtype=np.int64)] += (
            estimator.feature_importances_
        )
    importances /= len(model.estimators_)
    importance_frame = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_tree_importance": importances,
    }).sort_values("mean_tree_importance", ascending=False)
    importance_frame.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = importance_frame.sort_values("mean_tree_importance")
    ax.barh(ordered["feature"], ordered["mean_tree_importance"], color="#76B7B2")
    ax.set_xlabel("Mean impurity-based importance")
    ax.set_title("Bagging mean tree feature importance (descriptive)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=170)
    plt.close(fig)


def write_report(config, fold_frame, metrics, map_path):
    fold_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.training_trips)} | "
        f"{int(row.validation_trips)} | {row.validation_macro_f1:.4f} | "
        f"{row.training_macro_f1:.4f} | {row.fit_seconds:.1f} |"
        for row in fold_frame.itertuples(index=False)
    )
    report = f"""# Bagging model experiment

## Experiment scope

This is the bagging-only result. The model was fit and cross-validated on the 80 training trips using five group-aware folds; all rows from each trip stay in one fold. The ten `student_test` trips were used only for development-stage comparison. The reserved evaluator-testing partition is not present and was not used. No other model has been evaluated yet, so cross-model shortlisting and selection remain pending.

## Comparison-table row

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 accuracy | Student-test route continuity | Student-test Macro F1 | Student-test Log Loss | Inference time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Bagging | Bootstrap aggregation (tree ensemble) | {config['key_configuration']} | {metrics['cv_macro_f1_mean']:.4f} ± {metrics['cv_macro_f1_std']:.4f} | {metrics['student_test_top1_accuracy']:.4f} | {metrics['student_test_route_continuity']:.2%} ({metrics['route_continuity_connected_transitions']:,}/{metrics['route_continuity_eligible_transitions']:,}) | {metrics['student_test_candidate_macro_f1']:.4f} | {metrics['student_test_candidate_log_loss']:.4f} | {metrics['student_test_inference_seconds']:.2f} s |

The CV and student-test Macro F1 values are **candidate-level binary classification** metrics at threshold 0.5. Top-1 accuracy is **observation-level**: choose the candidate with the largest positive-class probability for each GPS fix and compare its label with the route-derived reference label. Log Loss is candidate-level and uses the averaged positive-class probabilities.

## Cross-validation results

| Fold | Training trips | Validation trips | Validation Macro F1 | Training Macro F1 | Fit time (s) |
|---:|---:|---:|---:|---:|---:|
{fold_rows}

Mean CV Macro F1 is {metrics['cv_macro_f1_mean']:.4f} (standard deviation {metrics['cv_macro_f1_std']:.4f}); the mean fold training Macro F1 is {metrics['cv_training_macro_f1_mean']:.4f}. Grouping by trip prevents candidate rows from one trip appearing in both the fit and validation parts of a fold. The fold scores are also plotted in `cv_macro_f1.png`.

## Student-testing results

- Candidate rows: {metrics['student_test_candidates']:,}; candidate positive rate: {metrics['student_test_candidate_positive_rate']:.2%}.
- Student-test trips and observations: {metrics['student_test_trips']} trips, {metrics['student_test_observations']:,} GPS observations.
- Candidate-level accuracy: {metrics['student_test_candidate_accuracy']:.4f}; macro precision: {metrics['student_test_candidate_macro_precision']:.4f}; macro recall: {metrics['student_test_candidate_macro_recall']:.4f}; macro F1: {metrics['student_test_candidate_macro_f1']:.4f}.
- Observation-level Top-1 accuracy: {metrics['student_test_top1_accuracy']:.4f} ({metrics['student_test_top1_correct']:,}/{metrics['student_test_observations']:,} observations).
- Top-1 accuracy among observations whose reference arc was present in the candidate set: {metrics['student_test_top1_accuracy_when_candidate_present']:.4f} ({metrics['student_test_top1_correct']:,}/{metrics['student_test_candidate_present_observations']:,}).
- Candidate-level Log Loss: {metrics['student_test_candidate_log_loss']:.4f}.
- Candidate-generation miss rate: {metrics['student_test_candidate_miss_rate']:.2%}; a reference candidate absent from the candidate set cannot be recovered by this classifier.
- For context, predicting every candidate as non-reference gives candidate accuracy {metrics['student_test_all_negative_accuracy']:.4f} but Macro F1 {metrics['student_test_all_negative_macro_f1']:.4f}. Bagging improves Macro F1 substantially, while its 0.5 threshold has lower raw accuracy than that majority-class baseline; accuracy alone is misleading for this imbalanced target.
- Strict directed route continuity: {metrics['student_test_route_continuity']:.2%} across {metrics['route_continuity_eligible_transitions']:,} adjacent observation pairs with gaps no longer than five seconds. Same-arc continuation or a direct directed arc connection counts as continuous; {metrics['route_continuity_long_gap_transitions_excluded']:,} longer-gap pairs are excluded.
- Final fit time on all training candidates: {metrics['final_fit_seconds']:.1f} seconds. Mean `predict_proba` inference time over three full student-test passes: {metrics['student_test_inference_seconds']:.2f} seconds for {metrics['student_test_candidates']:,} candidate rows ({metrics['student_test_inference_ms_per_candidate']:.4f} ms/candidate).

The CSV and JSON metrics are in this directory. The confusion matrix and metric plots use these actual held-out student-test predictions. `bagging_student_test_trip_{metrics['map_trip_id']}.html` overlays GPS observations, the supplied road network, reference route and Bagging top-1 predictions; `bagging_student_test_trip_{metrics['map_trip_id']}.png` is its screenshot for the technical report.

## Why consider Bagging for PathSnap?

The task has nonlinear interactions among spatial distance, segment position, motion and road-segment geometry. A bagged tree ensemble can model such interactions without feature scaling, while bootstrap aggregation reduces the variance of a single decision tree. Its averaged class probabilities can rank candidate segments and later serve as classifier scores for sequential decoding.

## Strengths and limitations for this task

**Strengths:** models nonlinear feature interactions; requires no standardization; can handle mixed feature scales; bootstrap aggregation is generally more stable than one unconstrained tree; supplies averaged probabilities; and fits naturally into the candidate-classification stage.

**Limitations:** the labels are derived from the supplied route and nearest-arc association, not independently verified ground truth. Candidate labels are imbalanced ({metrics['training_candidate_positive_rate']:.2%} positive in training); class weighting is therefore used, but probability calibration and threshold-dependent metrics still need scrutiny. Training and inference cost grow with the number and size of base trees. Independent candidate predictions do not enforce road-network connectivity or sequence consistency, so the route-continuity result is diagnostic rather than a decoder result. The strict one-hop continuity measure is conservative where a GPS interval may span multiple road segments.

## Configuration and training objective

| Setting | Initial value |
|---|---|
| Base estimator | `DecisionTreeClassifier` |
| Tree criterion | Gini impurity |
| Maximum tree depth | {config['max_depth']} |
| Minimum samples per leaf | {config['min_samples_leaf']} |
| Class weighting | `balanced` |
| Number of estimators | {config['n_estimators']} |
| Samples per estimator | {config['max_samples']} of the training fold, bootstrap sampling enabled |
| Features per estimator | {config['max_features']}; all nine features |
| CV | {CV_SPLITS}-fold `StratifiedGroupKFold`, grouped by trip ID |
| Random seed | {RANDOM_STATE} |

Each tree greedily minimizes class-weighted Gini impurity at its splits; bagging averages the resulting tree probabilities. There is no gradient-based loss or epoch-wise training history for this model. Model comparison uses CV Macro F1 as the primary candidate-classification criterion, alongside observation-level Top-1, student-test Macro F1, Log Loss, continuity and runtime. No artificial training/validation loss curve is reported.

## Overfitting assessment

{metrics['overfit_assessment']}

The final model's in-sample candidate Macro F1 is {metrics['training_candidate_macro_f1']:.4f}, compared with {metrics['cv_macro_f1_mean']:.4f} mean trip-grouped CV Macro F1 and {metrics['student_test_candidate_macro_f1']:.4f} student-test Macro F1. The in-sample-to-CV gap is {metrics['train_minus_cv_macro_f1']:.4f}; the CV-to-student-test gap is {metrics['cv_minus_student_test_macro_f1']:.4f}. The 0.05 Macro-F1 gap used for this descriptive warning is a practical flag, not a significance threshold. The held-out student-test result is for development comparison, not evaluator selection.

## Figures and next steps

- `cv_macro_f1.png`: actual grouped CV scores for Bagging (a single-model baseline, not yet a five-model comparison).
- `student_test_metrics.png`: observation Top-1 and candidate Macro F1, with their evaluation units labelled.
- `student_test_log_loss.png` and `student_test_confusion_matrix.png`: actual held-out predictions.
- `feature_importance.png`: descriptive mean tree impurity importance; correlated features can divide importance.
- `{map_path.name}`: actual Bagging prediction map.
- `bagging_student_test_trip_{metrics['map_trip_id']}.png`: screenshot for the technical report.

The deliverables require a consolidated comparison and plots across at least five trained models. Those comparisons and the final shortlist cannot be completed until the other model experiments are available. Bayesian hyperparameter optimisation should follow the initial five-model comparison; it is not claimed as completed here.
"""
    (OUTPUT_DIR / "bagging_results.md").write_text(report, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_frame, train_features, train_labels = load_partition("train")
    groups = train_frame["trip_id"].to_numpy(dtype=str)
    del train_frame
    gc.collect()
    test_frame, test_features, test_labels = load_partition("student_test")
    test_metadata = test_frame.loc[:, [*ID_COLUMNS, "label"]].copy()
    del test_frame
    gc.collect()

    print(
        f"Loaded {len(train_labels):,} training candidates from "
        f"{np.unique(groups).size} trips and "
        f"{len(test_labels):,} student-test candidates."
    )
    fold_frame = cross_validate(train_features, train_labels, groups)
    fold_frame.to_csv(OUTPUT_DIR / "cv_fold_metrics.csv", index=False)

    model = make_model(RANDOM_STATE)
    fit_start = time.perf_counter()
    model.fit(train_features, train_labels)
    final_fit_seconds = time.perf_counter() - fit_start
    train_predictions = model.predict(train_features)
    training_candidate_macro_f1 = float(f1_score(
        train_labels, train_predictions, average="macro", zero_division=0
    ))

    inference_times = []
    test_probabilities = None
    for _ in range(3):
        start = time.perf_counter()
        test_probabilities = positive_probability(model, test_features)
        inference_times.append(time.perf_counter() - start)
    assert test_probabilities is not None
    inference_seconds = float(np.mean(inference_times))
    test_predictions = (test_probabilities >= 0.5).astype(np.int8)
    precision, recall, _, _ = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    candidate_macro_f1 = float(f1_score(
        test_labels,
        test_predictions,
        average="macro",
        zero_division=0,
    ))
    top1, scored_candidates = top1_predictions(test_metadata, test_probabilities)
    route_metrics = measure_route_continuity(top1)
    unique_observations = int(top1.shape[0])
    top1_correct = int(top1["correct"].sum())
    observation_has_reference = test_metadata.groupby(
        ["trip_id", "observation_id"]
    )["label"].max()
    candidate_present_observations = int(observation_has_reference.sum())
    test_candidate_miss_rate = (
        1.0 - float(observation_has_reference.mean())
        if unique_observations
        else 0.0
    )
    confusion = confusion_matrix(test_labels, test_predictions, labels=[0, 1])
    all_negative = np.zeros_like(test_labels)
    majority_accuracy = float(accuracy_score(test_labels, all_negative))
    majority_macro_f1 = float(f1_score(
        test_labels, all_negative, average="macro", zero_division=0
    ))
    obs_counts = test_metadata.groupby("trip_id")["observation_id"].nunique()
    median_obs = float(obs_counts.median())
    map_trip_id = min(
        obs_counts.index,
        key=lambda trip_id: (abs(float(obs_counts[trip_id]) - median_obs), trip_id),
    )
    top1.to_csv(OUTPUT_DIR / "student_test_top1_predictions.csv", index=False)
    scored_candidates.to_csv(
        OUTPUT_DIR / "student_test_candidate_predictions.csv", index=False
    )

    metrics = {
        "model": "Bagging",
        "algorithm_family": "Bootstrap aggregation (tree ensemble)",
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "cv_grouping": "trip_id",
        "cv_macro_f1_mean": float(fold_frame["validation_macro_f1"].mean()),
        "cv_macro_f1_std": float(fold_frame["validation_macro_f1"].std(ddof=1)),
        "cv_training_macro_f1_mean": float(fold_frame["training_macro_f1"].mean()),
        "training_candidate_macro_f1": training_candidate_macro_f1,
        "training_candidate_positive_rate": float(train_labels.mean()),
        "student_test_trips": int(test_metadata["trip_id"].nunique()),
        "student_test_candidates": int(len(test_metadata)),
        "student_test_observations": unique_observations,
        "student_test_candidate_positive_rate": float(test_labels.mean()),
        "student_test_candidate_accuracy": float(
            accuracy_score(test_labels, test_predictions)
        ),
        "student_test_candidate_macro_precision": float(precision),
        "student_test_candidate_macro_recall": float(recall),
        "student_test_candidate_macro_f1": candidate_macro_f1,
        "student_test_candidate_log_loss": float(log_loss(
            test_labels, test_probabilities, labels=[0, 1]
        )),
        "student_test_candidate_miss_rate": float(test_candidate_miss_rate),
        "student_test_candidate_present_observations": candidate_present_observations,
        "student_test_top1_correct": top1_correct,
        "student_test_top1_accuracy": float(top1["correct"].mean()),
        "student_test_top1_accuracy_when_candidate_present": float(
            top1_correct / candidate_present_observations
            if candidate_present_observations
            else 0.0
        ),
        "student_test_all_negative_accuracy": majority_accuracy,
        "student_test_all_negative_macro_f1": majority_macro_f1,
        "student_test_route_continuity": route_metrics["continuity"],
        "route_continuity_connected_transitions": route_metrics["connected_transitions"],
        "route_continuity_eligible_transitions": route_metrics["eligible_transitions"],
        "route_continuity_long_gap_transitions_excluded": route_metrics[
            "long_gap_transitions_excluded"
        ],
        "route_continuity_definition": route_metrics["definition"],
        "final_fit_seconds": float(final_fit_seconds),
        "student_test_inference_seconds": inference_seconds,
        "student_test_inference_ms_per_candidate": float(
            inference_seconds * 1000 / len(test_metadata)
        ),
        "inference_repeats": len(inference_times),
        "inference_repeat_seconds": [float(value) for value in inference_times],
        "cv_total_fit_seconds": float(fold_frame["fit_seconds"].sum()),
        "train_minus_cv_macro_f1": float(
            training_candidate_macro_f1 - fold_frame["validation_macro_f1"].mean()
        ),
        "cv_minus_student_test_macro_f1": float(
            fold_frame["validation_macro_f1"].mean() - candidate_macro_f1
        ),
        "confusion_matrix_labels": ["non_reference", "reference"],
        "confusion_matrix": confusion.tolist(),
        "map_trip_id": str(map_trip_id),
    }
    config = {
        "estimator": "DecisionTreeClassifier",
        "criterion": "gini",
        "max_depth": 14,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "max_features": MAX_FEATURES,
        "bootstrap": True,
        "bootstrap_features": False,
        "n_jobs": N_JOBS,
        "random_state": RANDOM_STATE,
        "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "primary_cv_metric": "candidate-level macro F1 at threshold 0.5",
        "key_configuration": (
            "20 trees; depth 14; min leaf 20; balanced class weights; "
            "10% bootstrap sample per tree; all nine features"
        ),
    }
    metrics.update({
        "configuration": config,
        "candidate_threshold": 0.5,
        "student_test_accuracy_unit": "candidate",
        "student_test_macro_f1_unit": "candidate",
        "student_test_top1_accuracy_unit": "observation",
        "student_test_log_loss_unit": "candidate",
    })
    train_cv_gap = metrics["train_minus_cv_macro_f1"]
    if train_cv_gap >= 0.05:
        overfit_assessment = (
            "The in-sample score is at least 0.05 Macro-F1 above the grouped "
            "CV mean, which is a practical overfitting warning. Check the fold "
            "scores and held-out student-test gap before drawing a conclusion."
        )
    elif (
        metrics["training_candidate_macro_f1"] < 0.50
        and metrics["cv_macro_f1_mean"] < 0.50
    ):
        overfit_assessment = (
            "Both in-sample and grouped-CV Macro F1 are below 0.50, which is "
            "consistent with underfitting at this threshold/configuration; "
            "compare with the other model families before deciding."
        )
    else:
        overfit_assessment = (
            "There is no large in-sample-to-grouped-CV gap under the 0.05 "
            "practical flag, so these results show no strong overfitting "
            "signal; compare the student-test gap and other models as well."
        )
    metrics["overfit_assessment"] = overfit_assessment
    model_path = OUTPUT_DIR / "bagging_model.joblib"
    joblib.dump(model, model_path, compress=3)
    metrics["model_file"] = model_path.relative_to(ROOT).as_posix()
    map_path = create_prediction_map(str(map_trip_id), top1)
    metrics["prediction_map"] = map_path.relative_to(ROOT).as_posix()

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config_path = OUTPUT_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_row = {
        "model": "Bagging",
        "algorithm_family": metrics["algorithm_family"],
        "key_configuration": config["key_configuration"],
        "cv_macro_f1": metrics["cv_macro_f1_mean"],
        "cv_macro_f1_std": metrics["cv_macro_f1_std"],
        "student_test_top1_accuracy": metrics["student_test_top1_accuracy"],
        "student_test_route_continuity": metrics["student_test_route_continuity"],
        "student_test_candidate_macro_f1": metrics["student_test_candidate_macro_f1"],
        "student_test_candidate_log_loss": metrics["student_test_candidate_log_loss"],
        "student_test_inference_seconds": metrics["student_test_inference_seconds"],
        "final_fit_seconds": metrics["final_fit_seconds"],
    }
    pd.DataFrame([summary_row]).to_csv(
        OUTPUT_DIR / "comparison_table_row.csv", index=False
    )
    write_plots(fold_frame, metrics, confusion, model)
    write_report(config, fold_frame, metrics, map_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

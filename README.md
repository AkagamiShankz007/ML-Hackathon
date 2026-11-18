# PathSnap

Independent data-processing and feature-engineering pipeline for the PathSnap
map-matching dataset.

## Pipeline

Run:

```bash
python -m pip install -r requirements.txt
python src/build_features.py
python src/train_bagging.py
```

The pipeline:

1. Inventories every trip in `train/` and `student_test/`.
2. Removes rows with non-finite/out-of-range coordinates or non-finite timestamps,
   and removes consecutive exact duplicate observations. It preserves observation
   order and fails explicitly if timestamps decrease.
3. Preserves gaps over five seconds and records them; it does not interpolate
   across gaps because the vehicle's path during a dropout is unknown. Motion
   features are marked invalid at the first observation and after long gaps.
4. Converts longitude/latitude differences to approximate local metres.
5. Builds spatial candidates from road-segment geometry and uses the supplied
   route only to generate development labels, not as a model input feature.
6. Computes point-to-segment, motion, bearing, segment-length, and log-gap
   features.
7. Reports per-trip file presence/byte sizes and parsed input schema, calculates
   directed-road-network connectivity and route-arc continuity, and records
   thresholded, trip-specific anomalies.
8. Produces partition-labelled GPS trajectory, coordinate, observation-level
   time-gap, candidate-feature, and road-network plots.
9. Writes inventory, anomaly, route-quality, network, schema, partition,
   extent, feature, JSON, plot, findings-report, and Folium outputs under
   `artifacts/`.

The model-ready feature matrix and its `feature_names` are in
`artifacts/candidate_features.npz`. Reference labels and route-distance
diagnostics are separate target/diagnostic columns in the tabular exports and
must not be used as input features.

For CSV-based model selection, `model_selection_data/train.csv` and
`model_selection_data/student_test.csv` contain one partition per file. Use
only the nine feature columns as model inputs; `label` is the target, while
`trip_id`, `observation_id`, and `candidate_arc` are identifiers for grouping
and reconstructing predictions. Route-derived diagnostic columns are omitted
from these files to prevent target leakage. The generated CSVs are excluded
from Git because of their size.

The supplied package has no `evaluator_test/` directory. This is recorded as
unavailable in `artifacts/partition_inventory.csv`; no evaluator data or labels
are invented or read. The interactive exploration map includes sampled nearby
road segments as context, in addition to GPS observations and the reference
route.

The static report map screenshot is included at
`artifacts/dataset_exploration_map_screenshot.png`.

## Model selection: Bagging

`src/train_bagging.py` trains a bagged decision-tree candidate classifier. It
uses five-fold `StratifiedGroupKFold` cross-validation grouped by trip ID on the
80 training trips, then reports development metrics on the ten
`student_test` trips. It writes the actual fold scores, configuration, metrics,
saved model, comparison-table row, diagnostic plots, predictions, and a
Folium map under `artifacts/model_selection/bagging/`. The student-testing
partition is for development comparison only; the evaluator-testing partition
is unavailable and must not be used for model selection.
The map screenshot for the technical report is saved beside the HTML map.

The Bagging result is an initial model entry, not a final shortlist. The
five-model comparison charts and Bayesian hyperparameter search should be
completed only after the remaining model experiments are run.

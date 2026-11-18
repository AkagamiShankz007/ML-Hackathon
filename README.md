# PathSnap: map matching and route reconstruction

**Team:** Arham Burad (PES2UG24AM025), Harshad Sachin Patil (PES2UG24AM063), G Maqdhoom Pasha (PES2UG25AM801), Arjun Dil Kumar (PES2UG24AM026) - UE24CS352A Machine Learning, PES University.

PathSnap matches noisy GPS observations to a road network and reconstructs the route a vehicle travelled.
Final system: **Bayesian-tuned Logistic Regression** scores the 16 candidate road segments of every GPS fix, and an
**HMM / Viterbi decoder** turns those scores into a connected route. A FastAPI service runs the full pipeline on the raw
trip files; the RouteLens web UI displays the result.

| Student-test partition (10 trips) | Classifier only | Classifier + HMM/Viterbi |
|---|---|---|
| Observation Top-1 accuracy | 84.0 % | **87.3 %** |
| Route continuity (fixes <= 5 s apart) | 91.9 % | **99.5 %** |
| Route-arc F1 | 0.853 | **0.919** |

The numbers come from `artifacts/hmm/comparison.csv`; every figure and table of the report is generated from files in `artifacts/`.

## Run the system (two-laptop setup of the evaluation)

```bash
python -m pip install -r requirements.txt          # Python 3.11
```

**Laptop 1 - model and API** (hosts the classifier, the HMM/Viterbi decoder, the road network and the server-side evaluation):

```powershell
.\scripts\run_api.ps1 student_test            # or: ./scripts/run_api.sh student_test       -> http://<laptop-1-ip>:8000  (Swagger: /docs)
```

**Laptop 2 - user interface** (static files, talks to the API over HTTP):

```powershell
.\scripts\run_ui.ps1 <laptop-1-ip>            # or: ./scripts/run_ui.sh <laptop-1-ip>       -> http://localhost:5173/?api=http://<laptop-1-ip>:8000
```

Allow inbound TCP 8000 on laptop 1 (Windows Firewall).
Every request is logged to the console and `logs/api.log` (show the console to the evaluator).

* The API serves `<PATHSNAP_DATA_DIR>/<PATHSNAP_SERVE_PARTITION>/<trip>/<trip>.{track,nodes,arcs[,route]}`
  (defaults: `Student folder`, `student_test`). For the evaluation set `PATHSNAP_SERVE_PARTITION=evaluator_test` and, if the
  folder is elsewhere, `PATHSNAP_DATA_DIR`. Other variables: `PATHSNAP_MODEL_DIR` (default `artifacts/final_model`), `PATHSNAP_CORS_ORIGINS`.
* Endpoints: `GET /health`, `GET /trips`, `GET /trips/{trip_id}/network`, `POST /reconstruct {"trip_id", "mode": "hmm"|"classifier", "compare": bool}`.
* Each request cleans the raw track, generates candidates and features, scores them, decodes the route and computes the metrics.
  The `.route` file is read only inside the server for those metrics; no response field contains reference data
  (`scripts/check_api_all_trips.py` checks every trip and every response for this).
* `python scripts/check_api_all_trips.py` exercises all served trips through HTTP -> `artifacts/api_validation/all_trips_results.csv`.

## Repository layout

| Path | Content |
|---|---|
| `src/build_features.py` | raw-data inventory, cleaning, v1 candidate generation, labels, EDA plots/maps (shared pipeline) |
| `src/candidates_v2.py` | corrected candidate generation + direction features used by all final experiments (see "Data findings") |
| `src/logreg_common.py` | data loading, folds, label-quality filter shared by the final-model stages |
| `src/model_comparison.py`, `src/model_comparison_plots.py` | the unified comparison of 8 models (training, CV, student-test, plots, confusion matrices) |
| `artifacts/model_selection/<model>/`, `src/train_*.py`, `src/train_all_models.py` | per-model scripts and results from the first experiments on the v1 data (kept for the record) |
| `src/bayesian_opt_logistic.py` | Bayesian hyperparameter optimisation (Optuna TPE) of the final model |
| `src/hmm_decoder.py`, `src/hmm_tuning.py`, `src/compare_models_hmm.py` | HMM/Viterbi decoder, decoder tuning + classifier-vs-HMM evaluation, comparison across models |
| `src/export_final_model.py` | writes `artifacts/final_model/` (model, feature spec, HMM config, manifest) |
| `src/pathsnap_inference.py`, `api/main.py` | inference core and FastAPI service |
| `frontend/` | RouteLens evaluator UI (static HTML/JS, Leaflet vendored) |
| `scripts/` | run scripts (API/UI), API validation, Folium map and screenshot generators |
| `src/report_figures.py`, `report/` | report figures and the technical report (`report/PathSnap_Technical_Report.pdf`) |
| `artifacts/` | all recorded results: EDA, `model_comparison/`, `bayesian_optimization/`, `hmm/`, `final_model/`, `api_validation/`, `ui_screenshots/`, `report_figures/` |

## Reproduce the results

```bash
python src/build_features.py                 # v1 candidates + EDA artifacts (shared pipeline)
python src/candidates_v2.py                  # corrected candidates (10 s) -> artifacts/candidates_v2/ (not committed, regenerated)
python src/model_comparison.py               # 8 models: 5-fold trip-grouped CV + student-test -> artifacts/model_comparison/
python src/bayesian_opt_logistic.py --trials 30   # Bayesian optimisation of the final model -> artifacts/bayesian_optimization/
python src/hmm_tuning.py --trials 60         # decoder tuning, classifier vs classifier + HMM -> artifacts/hmm/
python src/compare_models_hmm.py             # classifier vs +HMM for every compared model
python src/model_comparison_plots.py         # consolidated table + comparison plots
python src/export_final_model.py             # artifacts/final_model/
python src/report_figures.py                 # figures for the data findings
```

All random seeds are 42. Folds: `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by trip on the 80 training trips.

## Data findings that shaped the pipeline

Found while analysing the errors of the first classifier + HMM (details and figures in the report):

1. **Candidate generation (v1) dropped the right road.** The 16 candidates were the arcs with the nearest *midpoint*; long arcs passing a few
   metres from the fix were missed (reference arc absent for 4.6 % of student-test and 9.1 % of training fixes; the missed arc was a median 6 m from the fix, the
   16th candidate ~80 m). v2 uses the exact point-to-segment distance: 0.03 % misses on student-test.
2. **`bearing_delta` (v1) could only be 0 or pi.** `np.angle` of a real number returns the sign only. v2 uses the wrapped absolute angle.
3. **Two training trips have corrupt labels** (`00000054`, `00000033`: reference route 120-140 m from the GPS track). They are excluded from fitting,
   cross-validation and decoder tuning (`logreg_common.bad_label_trips`); evaluation partitions are never filtered.
4. About two thirds of the remaining "wrong direction" errors are label ambiguity: the reference route traverses the road in both directions.

The shared files `src/build_features.py` and `artifacts/candidate_features.parquet` were not changed; v2 is generated by `src/candidates_v2.py`.

## Data partitions and evaluation integrity

Training (80 trips): fitting, cross-validation, Bayesian optimisation, decoder tuning. Student-test (10 trips): development comparison only.
Evaluator-test (10 trips): not present during development; never used for any choice. Preprocessing (scalers, standardisation) is fitted on training data only.
Reference routes never enter the prediction pipeline; the API reads them only to compute evaluation metrics on the server. No prediction is hard-coded.

---

# Original data-processing pipeline description (v1)

## PathSnap (v1 pipeline)

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


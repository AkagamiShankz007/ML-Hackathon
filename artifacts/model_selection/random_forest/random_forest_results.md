# Random Forest model experiment

## Experiment scope

This is the Random Forest-only result, produced with the same protocol as the Bagging folder: fit and cross-validated on the 80 training trips with five trip-grouped `StratifiedGroupKFold` folds (shuffle, seed 42); the ten `student_test` trips are used only for development comparison. The evaluator partition is not present and was not used. Thresholded metrics use 0.5.

**Feature-set note:** this model uses the extended 47-feature set built by `models/mlp/train_mlp.py` (the nine base columns plus within-fix relative-distance/rank features, distance to the same road segment in the fixes 1, 2, 3 and 5 steps before/after, and road-network consistency flags between neighbouring fixes' top candidates). None of them use labels. Models trained only on the nine base columns are therefore not like-for-like with these results.

## Comparison-table row

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 accuracy | Student-test route continuity | Student-test Macro F1 | Student-test Log Loss | Inference time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | Bagged decision-tree ensemble | 150 trees; depth 24; min leaf 10; 30% bootstrap sample per tree; sqrt features per split; Gini; 47 features | 0.8283 ± 0.0263 | 0.7347 | 91.46% (17,475/19,106) | 0.8613 | 0.0673 | 0.79 s |

CV and student-test Macro F1 are **candidate-level binary** metrics at threshold 0.5. Top-1 accuracy is **observation-level**: the candidate with the largest positive probability per GPS fix is compared with the route-derived reference label. Log Loss is candidate-level.

## Cross-validation results

| Fold | Training trips | Validation trips | Validation Macro F1 | Training Macro F1 | Validation Top-1 | Fit time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 16 | 0.8006 | 0.8902 | 0.4688 | 97.7 |
| 2 | 65 | 15 | 0.8240 | 0.9009 | 0.6725 | 93.7 |
| 3 | 65 | 15 | 0.8110 | 0.8995 | 0.6350 | 99.3 |
| 4 | 62 | 18 | 0.8683 | 0.8954 | 0.6927 | 97.5 |
| 5 | 64 | 16 | 0.8379 | 0.9000 | 0.6725 | 95.5 |

Mean CV Macro F1 is 0.8283 (std 0.0263); mean CV Top-1 is 0.6283 (std 0.0916). Top-1 varies between folds because trips differ in candidate miss rate and road-network density. Folds are trip-grouped, so no trip appears on both sides.

## Student-testing results

- Candidate rows: 306,560; positive rate: 5.96%; 10 trips, 19,160 GPS observations.
- Candidate-level accuracy: 0.9705; macro precision: 0.8818; macro recall: 0.8432; macro F1: 0.8613.
- Positive-class precision / recall / F1: 0.7826 / 0.6986 / 0.7382.
- Observation-level Top-1 accuracy: 0.7347 (14,077/19,160); among observations whose reference arc is in the candidate set: 0.7702 (14,077/18,277).
- Candidate-level Log Loss: 0.0673.
- Candidate-generation miss rate: 4.61% (candidate recall 95.39%); these cases cannot be recovered by any classifier.
- Strict directed route continuity: 91.46% across 19,106 adjacent observation pairs with gaps <=5 s (44 longer-gap pairs excluded). Classifier-only, no HMM.
- Final fit time on all training candidates: 118.0 s. Mean inference time over three full student-test passes: 0.79 s for 306,560 candidate rows (0.0026 ms/candidate).

Plots use the actual held-out predictions. `random_forest_student_test_trip_00000007.html` overlays GPS fixes, the road network, the reference route and the Random Forest top-1 predictions; the PNG is its screenshot for the report.

## Why consider Random Forest for PathSnap?

Candidate scoring depends on nonlinear threshold-like rules over distance, projection, direction and geometry; a forest of decorrelated deep trees captures these without scaling, is robust to outliers and gives vote-fraction probabilities that can rank candidates.

## Strengths and limitations for this task

**Strengths:** nonlinear interactions with no scaling; averaging many decorrelated trees stabilises predictions; trains and predicts on CPU; gives feature importances; probabilities are usable by the decoder.

**Limitations:** every candidate row is scored independently, so the forest cannot compare a candidate with the other 15 of the same fix (only the engineered relative features carry that information); vote-fraction probabilities are less calibrated than the MLP's (higher log loss); a larger model file and slower inference than the MLP; route-derived labels; candidate misses cannot be recovered.

## Configuration and training objective

Key configuration: 150 trees; depth 24; min leaf 10; 30% bootstrap sample per tree; sqrt features per split; Gini; 47 features. Full settings are in `config.json`. Random seed 42.

Each tree greedily minimises Gini impurity; the forest averages the trees' class probabilities. No gradient-based loss or training history.

## Overfitting assessment

The in-sample Macro F1 is 0.068 above the grouped CV mean (>=0.05 practical flag), a sign of overfitting; compare with the student-test gap below.

Mean CV Macro F1 0.8283 vs student-test Macro F1 0.8613 (gap -0.0329). Student-test is for development comparison, not evaluator selection.

## Reproduce

`python src/model_selection_export.py random_forest` (requires `artifacts/candidate_features.parquet` and `Student folder/`; the MLP and KNN stages use a CUDA GPU).

# Bagging model experiment

## Experiment scope

This is the bagging-only result. The model was fit and cross-validated on the 80 training trips using five group-aware folds; all rows from each trip stay in one fold. The ten `student_test` trips were used only for development-stage comparison. The reserved evaluator-testing partition is not present and was not used. No other model has been evaluated yet, so cross-model shortlisting and selection remain pending.

## Comparison-table row

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 accuracy | Student-test route continuity | Student-test Macro F1 | Student-test Log Loss | Inference time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Bagging | Bootstrap aggregation (tree ensemble) | 20 trees; depth 14; min leaf 20; balanced class weights; 10% bootstrap sample per tree; all nine features | 0.7725 ± 0.0070 | 0.5729 | 81.39% (15,551/19,106) | 0.7740 | 0.2054 | 0.43 s |

The CV and student-test Macro F1 values are **candidate-level binary classification** metrics at threshold 0.5. Top-1 accuracy is **observation-level**: choose the candidate with the largest positive-class probability for each GPS fix and compare its label with the route-derived reference label. Log Loss is candidate-level and uses the averaged positive-class probabilities.

## Cross-validation results

| Fold | Training trips | Validation trips | Validation Macro F1 | Training Macro F1 | Fit time (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 16 | 0.7757 | 0.7942 | 32.5 |
| 2 | 65 | 15 | 0.7624 | 0.7938 | 29.1 |
| 3 | 65 | 15 | 0.7683 | 0.7945 | 26.9 |
| 4 | 62 | 18 | 0.7793 | 0.7976 | 23.4 |
| 5 | 64 | 16 | 0.7770 | 0.7825 | 18.8 |

Mean CV Macro F1 is 0.7725 (standard deviation 0.0070); the mean fold training Macro F1 is 0.7925. Grouping by trip prevents candidate rows from one trip appearing in both the fit and validation parts of a fold. The fold scores are also plotted in `cv_macro_f1.png`.

## Student-testing results

- Candidate rows: 306,560; candidate positive rate: 5.96%.
- Student-test trips and observations: 10 trips, 19,160 GPS observations.
- Candidate-level accuracy: 0.9211; macro precision: 0.7125; macro recall: 0.9387; macro F1: 0.7740.
- Observation-level Top-1 accuracy: 0.5729 (10,977/19,160 observations).
- Top-1 accuracy among observations whose reference arc was present in the candidate set: 0.6006 (10,977/18,277).
- Candidate-level Log Loss: 0.2054.
- Candidate-generation miss rate: 4.61%; a reference candidate absent from the candidate set cannot be recovered by this classifier.
- For context, predicting every candidate as non-reference gives candidate accuracy 0.9404 but Macro F1 0.4846. Bagging improves Macro F1 substantially, while its 0.5 threshold has lower raw accuracy than that majority-class baseline; accuracy alone is misleading for this imbalanced target.
- Strict directed route continuity: 81.39% across 19,106 adjacent observation pairs with gaps no longer than five seconds. Same-arc continuation or a direct directed arc connection counts as continuous; 44 longer-gap pairs are excluded.
- Final fit time on all training candidates: 47.2 seconds. Mean `predict_proba` inference time over three full student-test passes: 0.43 seconds for 306,560 candidate rows (0.0014 ms/candidate).

The CSV and JSON metrics are in this directory. The confusion matrix and metric plots use these actual held-out student-test predictions. `bagging_student_test_trip_00000007.html` overlays GPS observations, the supplied road network, reference route and Bagging top-1 predictions; `bagging_student_test_trip_00000007.png` is its screenshot for the technical report.

## Why consider Bagging for PathSnap?

The task has nonlinear interactions among spatial distance, segment position, motion and road-segment geometry. A bagged tree ensemble can model such interactions without feature scaling, while bootstrap aggregation reduces the variance of a single decision tree. Its averaged class probabilities can rank candidate segments and later serve as classifier scores for sequential decoding.

## Strengths and limitations for this task

**Strengths:** models nonlinear feature interactions; requires no standardization; can handle mixed feature scales; bootstrap aggregation is generally more stable than one unconstrained tree; supplies averaged probabilities; and fits naturally into the candidate-classification stage.

**Limitations:** the labels are derived from the supplied route and nearest-arc association, not independently verified ground truth. Candidate labels are imbalanced (5.68% positive in training); class weighting is therefore used, but probability calibration and threshold-dependent metrics still need scrutiny. Training and inference cost grow with the number and size of base trees. Independent candidate predictions do not enforce road-network connectivity or sequence consistency, so the route-continuity result is diagnostic rather than a decoder result. The strict one-hop continuity measure is conservative where a GPS interval may span multiple road segments.

## Configuration and training objective

| Setting | Initial value |
|---|---|
| Base estimator | `DecisionTreeClassifier` |
| Tree criterion | Gini impurity |
| Maximum tree depth | 14 |
| Minimum samples per leaf | 20 |
| Class weighting | `balanced` |
| Number of estimators | 20 |
| Samples per estimator | 0.1 of the training fold, bootstrap sampling enabled |
| Features per estimator | 1.0; all nine features |
| CV | 5-fold `StratifiedGroupKFold`, grouped by trip ID |
| Random seed | 42 |

Each tree greedily minimizes class-weighted Gini impurity at its splits; bagging averages the resulting tree probabilities. There is no gradient-based loss or epoch-wise training history for this model. Model comparison uses CV Macro F1 as the primary candidate-classification criterion, alongside observation-level Top-1, student-test Macro F1, Log Loss, continuity and runtime. No artificial training/validation loss curve is reported.

## Overfitting assessment

There is no large in-sample-to-grouped-CV gap under the 0.05 practical flag, so these results show no strong overfitting signal; compare the student-test gap and other models as well.

The final model's in-sample candidate Macro F1 is 0.7916, compared with 0.7725 mean trip-grouped CV Macro F1 and 0.7740 student-test Macro F1. The in-sample-to-CV gap is 0.0191; the CV-to-student-test gap is -0.0015. The 0.05 Macro-F1 gap used for this descriptive warning is a practical flag, not a significance threshold. The held-out student-test result is for development comparison, not evaluator selection.

## Figures and next steps

- `cv_macro_f1.png`: actual grouped CV scores for Bagging (a single-model baseline, not yet a five-model comparison).
- `student_test_metrics.png`: observation Top-1 and candidate Macro F1, with their evaluation units labelled.
- `student_test_log_loss.png` and `student_test_confusion_matrix.png`: actual held-out predictions.
- `feature_importance.png`: descriptive mean tree impurity importance; correlated features can divide importance.
- `bagging_student_test_trip_00000007.html`: actual Bagging prediction map.
- `bagging_student_test_trip_00000007.png`: screenshot for the technical report.

The deliverables require a consolidated comparison and plots across at least five trained models. Those comparisons and the final shortlist cannot be completed until the other model experiments are available. Bayesian hyperparameter optimisation should follow the initial five-model comparison; it is not claimed as completed here.

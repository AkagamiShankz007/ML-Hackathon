# MLP model experiment

## Experiment scope

This is the MLP-only result, produced with the same protocol as the Bagging folder: fit and cross-validated on the 80 training trips with five trip-grouped `StratifiedGroupKFold` folds (shuffle, seed 42); the ten `student_test` trips are used only for development comparison. The evaluator partition is not present and was not used. Thresholded metrics use 0.5.

**Feature-set note:** this model uses the extended 47-feature set built by `models/mlp/train_mlp.py` (the nine base columns plus within-fix relative-distance/rank features, distance to the same road segment in the fixes 1, 2, 3 and 5 steps before/after, and road-network consistency flags between neighbouring fixes' top candidates). None of them use labels. Models trained only on the nine base columns are therefore not like-for-like with these results.

## Comparison-table row

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 accuracy | Student-test route continuity | Student-test Macro F1 | Student-test Log Loss | Inference time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MLP | Neural network (listwise candidate MLP) | 4 residual blocks, width 256, SiLU+LayerNorm, dropout 0.3, max-pool context over the 16 candidates; AdamW lr 0.002 (OneCycle), wd 0.01, batch 512, 12 epochs; BCE + listwise softmax loss; 47 features | 0.8962 ± 0.0266 | 0.8148 | 92.39% (17,652/19,106) | 0.9184 | 0.0536 | 1.21 s |

CV and student-test Macro F1 are **candidate-level binary** metrics at threshold 0.5. Top-1 accuracy is **observation-level**: the candidate with the largest ranking-head score per GPS fix is compared with the route-derived reference label. Log Loss is candidate-level. For the MLP, Top-1 with the probability head instead of the ranking head is 0.8171.

## Cross-validation results

| Fold | Training trips | Validation trips | Validation Macro F1 | Training Macro F1 | Validation Top-1 | Fit time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 16 | 0.8548 | 0.9314 | 0.5231 | 101.1 |
| 2 | 65 | 15 | 0.9026 | 0.9462 | 0.7990 | 92.6 |
| 3 | 65 | 15 | 0.8868 | 0.9454 | 0.7681 | 97.4 |
| 4 | 62 | 18 | 0.9198 | 0.9533 | 0.7733 | 92.7 |
| 5 | 64 | 16 | 0.9170 | 0.9419 | 0.7999 | 76.0 |

Mean CV Macro F1 is 0.8962 (std 0.0266); mean CV Top-1 is 0.7327 (std 0.1180). Top-1 varies between folds because trips differ in candidate miss rate and road-network density. Folds are trip-grouped, so no trip appears on both sides.

## Student-testing results

- Candidate rows: 306,560; positive rate: 5.96%; 10 trips, 19,160 GPS observations.
- Candidate-level accuracy: 0.9814; macro precision: 0.9117; macro recall: 0.9254; macro F1: 0.9184.
- Positive-class precision / recall / F1: 0.8322 / 0.8618 / 0.8468.
- Observation-level Top-1 accuracy: 0.8148 (15,612/19,160); among observations whose reference arc is in the candidate set: 0.8542 (15,612/18,277).
- Candidate-level Log Loss: 0.0536.
- Candidate-generation miss rate: 4.61% (candidate recall 95.39%); these cases cannot be recovered by any classifier.
- Strict directed route continuity: 92.39% across 19,106 adjacent observation pairs with gaps <=5 s (44 longer-gap pairs excluded). Classifier-only, no HMM.
- Final fit time on all training candidates: 108.3 s. Mean inference time over three full student-test passes: 1.21 s for 306,560 candidate rows (0.0039 ms/candidate).

Plots use the actual held-out predictions. `mlp_student_test_trip_00000007.html` overlays GPS fixes, the road network, the reference route and the MLP top-1 predictions; the PNG is its screenshot for the report.

## Why consider MLP for PathSnap?

Each GPS fix has 16 competing candidate segments, and the right choice depends on interactions between distance, direction, segment geometry, speed and the neighbouring fixes. An MLP can learn those nonlinear interactions, and because it is differentiable it can be trained with a ranking (listwise softmax) loss that targets Top-1 directly alongside a calibrated binary head whose probabilities feed the HMM.

## Strengths and limitations for this task

**Strengths:** learns nonlinear feature interactions; the context pooling lets every candidate see the other 15 candidates of its fix, which plain per-row classifiers cannot do; two heads give both calibrated probabilities (BCE) and a ranking score; inference is fast on GPU; probabilities plug directly into the HMM emission model.

**Limitations:** needs standardised inputs and a GPU for comfortable training; sensitive to hyper-parameters (dropout, weight decay, epochs) and seed; less interpretable than trees; overfits within a few epochs without regularisation and early stopping; the labels are route-derived, not independently verified; candidate-generation misses cannot be recovered.

## Configuration and training objective

Key configuration: 4 residual blocks, width 256, SiLU+LayerNorm, dropout 0.3, max-pool context over the 16 candidates; AdamW lr 0.002 (OneCycle), wd 0.01, batch 512, 12 epochs; BCE + listwise softmax loss; 47 features. Full settings are in `config.json`. Random seed 42.

Loss = binary cross-entropy on the probability head + 1.0 x listwise softmax cross-entropy on the ranking head (target distribution = reference candidate). Optimiser AdamW with a OneCycle schedule, bf16 autocast, early stopping on validation Top-1 using 1/8 of the training trips held out inside each fit.

## Overfitting assessment

The in-sample Macro F1 is 0.057 above the grouped CV mean (>=0.05 practical flag), a sign of overfitting; compare with the student-test gap below. Training loss keeps falling after roughly epoch 8 while validation Top-1 and log loss flatten (see `training_curves.png`), which is why early stopping and dropout 0.3 / weight decay 0.01 are used.

Mean CV Macro F1 0.8962 vs student-test Macro F1 0.9184 (gap -0.0222). Student-test is for development comparison, not evaluator selection.

## Reproduce

`python src/model_selection_export.py mlp` (requires `artifacts/candidate_features.parquet` and `Student folder/`; the MLP and KNN stages use a CUDA GPU).

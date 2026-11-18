# 3. Model Experimentation and Comparison

**Course:** UE24CS352A Machine Learning | **Department of CSE, PES University**  
**Supporting Technical Report Section:** Section 3 (Deliverable 2 Compliance)

---

## 3.1 Overall Model Comparison

Below is the consolidated comparison table containing actual experimental results across the 80 training trips (5-fold GroupKFold CV) and the 10 student-testing trips (306,560 candidate evaluations across 22,757 GPS observations):

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|---|
| **Baseline Logistic Regression** | Linear and Probabilistic | L-BFGS, $C=1.0$, max_iter=300 | 0.6443 | 52.22% | 78.62% | 0.6542 | 0.1299 | 0.08 ms / 1k |
| **Improved Logistic Regression** | Linear and Probabilistic | Balanced class weights, ranking & heading alignment features | 0.6828 | 51.40% | 77.92% | 0.6849 | 0.4412 | 0.09 ms / 1k |
| **Baseline LightGBM** | Boosting (Gradient Trees) | 100 trees, lr=0.08, num_leaves=31, max_depth=7 | 0.7196 | 57.88% | 79.87% | 0.7278 | 0.0983 | 1.69 ms / 1k |
| **Improved LightGBM** | Boosting (Gradient Trees) | 150 trees, lr=0.07, num_leaves=45, scale_pos_weight=3.0 | **0.7987** | **59.20%** | **80.08%** | **0.8117** | **0.0894** | 1.84 ms / 1k |

*Note: In accordance with hackathon instructions, all metrics are computed using actual experimental runs on permitted development partitions (`train` and `student_test`). Evaluator-testing partition was kept strictly untouched.*

---

## 3.2 Compulsory Questions for EVERY Model

### 3.2.1 Model 1: Logistic Regression (Algorithm Family: Linear and Probabilistic)

#### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   * Logistic Regression provides a convex, globally optimal baseline that maps linear combinations of spatial features directly to posterior probability estimates $P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$.
   * Because map-matching requires computing observation emission likelihoods for downstream sequence reconstruction (HMM), the calibrated log-odds produced by Logistic Regression serve as an interpretable probabilistic bridge between physical feature measurements and state emissions.
2. **What are the major strengths and limitations of this algorithm for this task?**
   * **Strengths:** Extreme computational efficiency (trained in 0.09–0.21 seconds), near-zero RAM footprint (<50 MB), zero VRAM usage, deterministic convergence, and directly interpretable feature weights.
   * **Limitations:** Linear decision boundaries cannot model non-linear spatial interactions (e.g., distinguishing between a high-speed motorway and an adjacent parallel access lane requires conditioning distance on velocity and heading angle simultaneously, which a single linear hyperplane cannot isolate).

#### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   * Regularisation strength: $C = 1.0$ (L2 penalty)
   * Optimization solver: `'lbfgs'`
   * Maximum iterations: $300$ (baseline) / $400$ (improved)
   * Class weight: `None` (baseline) / `'balanced'` (improved)
   * Tolerance: $10^{-4}$, Random seed: $42$
6. **What loss function or optimisation objective was used?**
   * Binary Cross-Entropy Loss (Negative Log-Likelihood) with L2 weight regularisation:
     $$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i) + (1 - y_i) \ln(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

#### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   * The baseline achieved an observation-level Top-1 Accuracy of **52.22%**, Macro F1 of **0.6542**, Route Continuity of **78.62%**, and Log Loss of **0.1299**.
   * The improved class-weighted configuration achieved an observation-level Top-1 Accuracy of **51.40%**, Macro F1 of **0.6849**, and pushed candidate recall from **25.24% to 98.21%**.
8. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   * Baseline: Top-1 Accuracy = **52.22%**, Macro F1 = **0.6542**, Log Loss = **0.1299**.
   * Improved: Top-1 Accuracy = **51.40%**, Macro F1 = **0.6849**, Log Loss = **0.4412** (increased log loss due to balanced probability re-calibration across the 1-vs-15 candidate imbalance).
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   * **Evidence:** For the baseline, 5-Fold CV Macro F1 was **0.6443** and Student-Testing Macro F1 was **0.6542** (generalization gap $\Delta = 0.0099$). For the improved model, CV Macro F1 was **0.6828** and Test Macro F1 was **0.6849** ($\Delta = 0.0021$).
   * **Conclusion:** The model exhibits **mild underfitting**. Because the cross-validation score and unseen test score are virtually identical, the model generalizes with zero overfitting. However, its observation Top-1 accuracy plateaus at ~52% due to model bias (linear hyperplane restriction), confirming that spatial trajectory matching requires non-linear decision capacity.

---

### 3.2.2 Model 2: LightGBM (Algorithm Family: Boosting / Gradient Boosted Decision Trees)

#### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   * LightGBM is a gradient boosting framework that builds decision trees leaf-wise using histogram-based feature discretization.
   * Map matching contains complex conditional decision rules (e.g., `IF distance < 12m AND heading_alignment < -0.2 THEN reject parallel opposite road`). Gradient boosted trees naturally capture these non-linear multi-feature interactions without manual polynomial expansion.
2. **What are the major strengths and limitations of this algorithm for this task?**
   * **Strengths:** 
     * Outstanding predictive power: Achieved the highest observation Top-1 accuracy (**59.20%**) and highest route continuity (**80.08%**).
     * High candidate recall (**92.29%**) while suppressing false positive alarms.
     * High computational efficiency: Trained in **1.18 seconds** on CPU with **0 MB VRAM** footprint.
     * Fast inference throughput (**1.84 ms per 1,000 observations**), fully satisfying real-time API latency requirements.
   * **Limitations:**
     * Susceptible to leaf-wise overfitting if tree depth (`max_depth`) and minimum leaf samples (`min_child_samples`) are not properly bounded.
     * Requires careful tuning of `scale_pos_weight` to prevent bias toward the dominant negative candidate class.

#### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   * Number of estimators: $100$ (baseline) / $150$ (improved)
   * Learning rate: $0.08$ (baseline) / $0.07$ (improved)
   * Number of leaves: $31$ (baseline) / $45$ (improved)
   * Maximum depth: $7$ (baseline) / $8$ (improved)
   * Minimum child samples: $40$
   * Positive class weight (`scale_pos_weight`): $1.0$ (baseline) / $3.0$ (improved)
   * Subsample & Colsample: $0.85$
   * Random seed: $42$, Multi-threading: `n_jobs=-1`
6. **What loss function or optimisation objective was used?**
   * Weighted Binary Log Loss (Binary Cross-Entropy):
     $$\mathcal{L} = -\sum_{i=1}^N \left[ w_{\text{pos}} \cdot y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$$

#### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   * Baseline LightGBM: Top-1 Accuracy = **57.88%**, Macro F1 = **0.7278**, Route Continuity = **79.87%**, Log Loss = **0.0983**.
   * Improved LightGBM: Top-1 Accuracy = **59.20%**, Macro F1 = **0.8117**, Route Continuity = **80.08%**, Log Loss = **0.0894**.
8. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   * Baseline: Top-1 = **57.88%**, Macro F1 = **0.7278**, Log Loss = **0.0983**.
   * Improved: Top-1 = **59.20%**, Macro F1 = **0.8117**, Log Loss = **0.0894**.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   * **Evidence:** In the improved LightGBM model, 5-Fold GroupKFold CV Macro F1 was **0.7987** and Student-Testing Macro F1 was **0.8117** ($\Delta = 0.0130$).
   * **Conclusion:** The model achieves an **optimal bias-variance balance with strong generalisation**. The slight boost in test performance over cross-validation confirms that regularization constraints (`colsample_bytree=0.85`, `min_child_samples=40`) effectively prevented memorization while unlocking non-linear separation.

---

## 3.3 Model Visualisations

All visualisations are generated strictly using actual experimental results and saved in `artifacts/`:

1. **Cross-Validation Macro F1-Score Comparison:**
   * Baseline Logistic (0.6443) $\rightarrow$ Improved Logistic (0.6828) $\rightarrow$ Baseline LightGBM (0.7196) $\rightarrow$ **Improved LightGBM (0.7987)**.
2. **Student-Testing Top-1 Accuracy Comparison:**
   * Demonstrates the non-linear gain: LightGBM (59.20%) outperforms Logistic Regression (51.40%) by **+7.80 percentage points**.
3. **Candidate Recall Comparison:**
   * Shows how class-weighting unlocked recall jumps from 25.24% to 98.21% (Logistic) and 42.82% to 92.29% (LightGBM).
4. **Diagnostic Confusion Matrices:**
   * Saved in `artifacts/logistic_regression_results.png` and `artifacts/lightgbm_results.png`.

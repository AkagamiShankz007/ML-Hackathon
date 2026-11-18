# Section 3.2: Model Evaluation Report — Logistic Regression

**Algorithm Family:** Linear and Probabilistic  
**Task:** Candidate Road Segment Classification & Observation-Level Map Matching  
**Dataset Partitions:**
* **Training Partition:** 80 Trips (3,334,672 candidate observations, evaluated via 5-Fold GroupKFold Cross-Validation)
* **Student-Testing Partition:** 10 Blind Trips (306,560 candidate observations, observation-level & candidate-level evaluation)

---

## A. Model Understanding

### 1. Why did you consider this algorithm suitable for the PathSnap problem?
Logistic Regression was chosen as the primary linear and probabilistic baseline for the PathSnap map-matching problem for the following core reasons:
1. **Direct Probabilistic Output for Emission Modeling:** In map matching, each GPS observation generates candidate road segments that must be scored. Under Hidden Markov Model (HMM) formulation, the candidate score serves directly as the observation emission probability $P(z_t \mid s_t)$. Logistic Regression uses the sigmoid function $\sigma(\mathbf{w}^T \mathbf{x} + b)$ to naturally output smooth, well-calibrated probabilities bounded strictly in $[0, 1]$.
2. **Convex Optimization with Zero Local Minima:** Logistic Regression minimizes convex binary cross-entropy, ensuring globally optimal parameter convergence without stochastic initialization variance.
3. **Rigorous Benchmark for Linear Separability:** As mandated by course guidelines, baseline models establish the performance lower-bound. By training Logistic Regression, we isolate whether spatial features (orthogonal distance, heading delta, projection, velocity) are linearly separable or whether non-linear boundary partitioning is required.
4. **Computational Efficiency for Distributed Deployment:** Logistic Regression requires near-zero CPU memory (< 50 MB RAM, 0 MB VRAM) and provides ultra-fast inference throughput (sub-millisecond per 1,000 candidates), making it an ideal candidate for low-resource edge servers.

---

### 2. What are the major strengths and limitations of this algorithm for this task?

#### Major Strengths:
* **Interpretability:** The learned weight vector $\mathbf{w}$ gives direct, physical insight into how each spatial and kinematic feature impacts road selection. Feature signs and magnitudes directly validate physical domain intuition.
* **Extremely Fast Convergence:** Solves via L-BFGS in **0.09 to 0.21 seconds** on hundreds of thousands of candidate observations.
* **Low Memory Footprint:** The serialized model artifact is under 1 KB (`.joblib` size: ~900 bytes), with zero GPU dependencies.
* **High Candidate Rejection Specificity:** Correctly rejects **283,971 out of 288,283 non-traversed road candidates** (> 98.5% True Negative rate in baseline configuration).

#### Major Limitations:
* **Linear Hypothesis Boundary:** Logistic Regression constructs planar decision boundaries. It cannot learn non-linear feature interactions (such as the joint conditional logic: *"if vehicle speed is high AND bearing delta exceeds 60°, candidate is an opposing highway lane regardless of distance"*).
* **Sensitivity to Extreme Class Imbalance:** Traversed road segments constitute only ~5.9% of candidate samples (1:16 ratio). Standard maximum likelihood estimation causes the decision threshold to favor the majority negative class, leading to suppressed recall (25.24%) unless class weighting or ranking transformations are applied.
* **Pointwise Evaluation without In-Observation Ranking:** Standard logistic regression treats every candidate row independently rather than conditioning on the relative rank of candidates belonging to the same observation timestamp.

---

## B. Training and Configuration

### 5. What were the important hyperparameters and their initial values?

Two distinct configurations were trained and evaluated:

#### Configuration 1: Baseline L2-Regularized Logistic Regression
* **Regularization Strength ($C$):** `1.0` (Inverse regularization parameter; controls L2 penalty)
* **Penalty:** `l2` (Ridge penalty: $\frac{1}{2} \|\mathbf{w}\|_2^2$)
* **Optimization Solver:** `'lbfgs'` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno quasi-Newton method)
* **Maximum Iterations:** `300`
* **Tolerance (`tol`):** `1e-4`
* **Class Weighting:** `None` (Standard empirical distribution weighting)
* **Feature Preprocessing:** `StandardScaler` (Z-score standardization: zero mean, unit variance across all 9 features)
* **Random Seed:** `42`

#### Configuration 2: Improved Class-Weighted + Rank-Normalized Logistic Regression
* **Class Weighting (`class_weight`):** `'balanced'` (Automatically adjusts weights inversely proportional to class frequencies: $w_j = \frac{N}{2 \cdot N_j}$)
* **Regularization Strength ($C$):** `0.5` (Slightly stronger L2 penalty to control coefficient variance with balanced weighting)
* **Engineered Input Dimensions:** 12 features (added `obs_dist_rank`, `heading_alignment = cos(bearing_delta)`, and `speed_aligned = speed * cos(bearing_delta)`)
* **Optimization Solver:** `'lbfgs'`
* **Maximum Iterations:** `500`

---

### 6. What loss function or optimisation objective was used?

The model optimizes the **L2-Regularized Binary Cross-Entropy Loss** (Negative Log-Likelihood):

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i + b) + (1 - y_i) \ln(1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b)) \right] + \frac{1}{2C} \sum_{j=1}^D w_j^2$$

Where:
* $y_i \in \{0, 1\}$ is the ground-truth binary label (1 = traversed arc, 0 = non-traversed candidate arc).
* $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function producing posterior class probability.
* $C$ controls the trade-off between fitting training data log-likelihood and penalizing large coefficient weights.
* In the improved configuration, sample weights $s_i$ are incorporated: $\mathcal{L}_{\text{weighted}} = -\frac{1}{N} \sum_{i=1}^N s_i \left[ y_i \ln \sigma(z_i) + (1 - y_i) \ln(1 - \sigma(z_i)) \right]$.

---

## C. Experimental Results

### 7. What was the model's performance on the student-testing dataset?

The models were evaluated on the blind **Student-Testing Partition** consisting of **10 complete trips** (306,560 candidate road arc rows).

#### Consolidated Metrics Summary:

| Evaluation Metric | Baseline Logistic Regression | Improved Logistic Regression |
| :--- | :---: | :---: |
| **5-Fold GroupKFold CV Macro F1-score** | **0.6443** ($\pm 0.0237$) | **0.6828** ($\pm 0.0210$) |
| **Student-Testing Top-1 Accuracy** | **52.22%** | **51.40%** |
| **Route Continuity (Classifier-Only)** | **78.62%** | **77.92%** |
| **Student-Testing Macro F1-score** | **0.6542** | **0.6849** |
| **Precision (Traversed Class 1)** | 0.5169 | 0.2931 |
| **Recall (Traversed Class 1)** | 25.24% | **98.21%** |
| **Log Loss (Cross-Entropy)** | **0.1299** | 0.3231 |
| **Total Training Time (CPU)** | **0.09 s** | **0.21 s** |
| **Inference Throughput** | **0.08 ms / 1k** | **0.15 ms / 1k** |

---

### 8. What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?

* **Observation-level Top-1 Accuracy:**
  * **Baseline:** **52.22%**
  * **Improved:** **51.40%**
  * *Meaning:* For over half of all GPS observations, the single highest-probability candidate predicted by the linear model corresponds to the true traversed road segment.
* **Macro F1-score:**
  * **Baseline:** **0.6542**
  * **Improved:** **0.6849**
  * *Meaning:* Balanced harmonic mean between traversed and non-traversed classes across 306,560 candidates.
* **Log Loss (Cross-Entropy):**
  * **Baseline:** **0.1299**
  * **Improved:** **0.3231**
  * *Meaning:* The baseline model achieves very low cross-entropy error because its unweighted probabilities reflect the true prior base rate (~5.9%).

#### Detailed Per-Trip Student-Testing Breakdown (10 Trips):

```text
-----------------------------------------------------------------------------------------
Trip ID      | GPS Points   | Top-1 Accuracy (Baseline) | Route Continuity (Baseline)
-----------------------------------------------------------------------------------------
00000000     | 2,503        | 47.30%                    | 88.97%
00000007     | 1,354        | 51.70%                    | 90.98%
00000009     | 1,049        | 37.56%                    | 64.89%
00000040     | 2,995        | 54.76%                    | 87.04%
00000052     | 262          | 74.43%                    | 77.39%
00000056     | 3,958        | 61.19%                    | 74.48%
00000065     | 768          | 57.55%                    | 75.49%
00000076     | 1,486        | 39.97%                    | 73.00%
00000077     | 3,911        | 49.71%                    | 81.69%
00000091     | 874          | 56.18%                    | 72.28%
-----------------------------------------------------------------------------------------
Average      | Total: 18,860| 52.22%                    | 78.62%
-----------------------------------------------------------------------------------------
```

---

### 9. Did you observe overfitting or underfitting? What evidence supports your conclusion?

* **Verdict:** **Clear Underfitting (High Bias, Low Variance).**
* **Empirical Evidence Supporting Conclusion:**
  1. **Minimal Generalization Gap ($\Delta \approx 0.01$):**
     * 5-Fold Cross-Validation Macro F1: **0.6443**
     * Student-Testing Macro F1: **0.6542**
     * The model generalizes consistently across unseen trips, with no drop in test performance. This confirms **zero overfitting**.
  2. **Performance Ceiling Due to Linear Restriction:**
     * Despite training on over 3.3 million candidate pairs, observation-level Top-1 accuracy plateaus at ~52.2%.
     * When class weights were balanced to improve recall (rising to 98.21%), precision dropped to 29.31% and false alarms rose to 43,280.
     * The linear hyperplane cannot isolate narrow, curved road intersections or distinguish close parallel streets without complex non-linear decision splits.

---

## 3.3 Model Visualisations & Diagnostics

### A. Confusion Matrix Breakdown (Student-Test: 306,560 Candidates)

#### Baseline Logistic Regression:
* **True Negatives (TN):** `283,971` (Rejected false candidates)
* **False Positives (FP):** `4,312` (False alarms)
* **False Negatives (FN):** `13,664` (Missed ground-truth arcs)
* **True Positives (TP):** `4,613` (Correctly identified arcs)

#### Improved Logistic Regression (Balanced):
* **True Negatives (TN):** `245,003`
* **False Positives (FP):** `43,280`
* **False Negatives (FN):** `328` (Nearly eliminated false negatives)
* **True Positives (TP):** `17,949`

### B. Learned Feature Coefficients ($\mathbf{w}$)
Analysis of learned standardized feature weights reveals direct physical interpretability:

1. **`log_distance` ($-1.7649$ / $-1.0710$):** Largest negative weight. Candidate likelihood decays exponentially with perpendicular GPS-to-road distance.
2. **`log_segment_length` ($+0.5733$):** Positive coefficient; captures longer vehicle dwell time on continuous arterials.
3. **`projection` ($+0.1839$ / $+0.4681$):** Positive weight; rewards candidates where the orthogonal projection falls within segment limits ($\alpha \in [0, 1]$).
4. **`obs_dist_rank` ($-2.5182$ in Improved Model):** Heavily penalizes candidates that are not the closest road to the observation.
5. **`bearing_delta` ($-0.0385$):** Penalizes angular divergence between vehicle heading and road arc orientation.

*Artifacts saved:*
* Model: `model_selection/logistic_regression/logistic_regression.joblib`
* Scaler: `model_selection/logistic_regression/scaler_logistic.joblib`
* Plot: `model_selection/logistic_regression/logistic_regression_results.png`
* Metrics: `model_selection/logistic_regression/logistic_regression_metrics.csv`

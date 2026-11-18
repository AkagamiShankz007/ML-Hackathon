# Model 1 Evaluation Report: Logistic Regression

**Algorithm Family:** Linear and Probabilistic  
**Task:** Candidate Road Segment Classification & Observation-Level Map Matching  
**Partition Evaluation:** 80 Training Trips (5-Fold GroupKFold CV) | 10 Student-Testing Trips (Blind Dev Evaluation)

---

## 1. Model Understanding (Section 3.2.A)

### 1.1 Why is this algorithm suitable for PathSnap?
Logistic Regression provides a foundational, convex probabilistic classifier. In map-matching pipelines, candidate road segments must be ranked for each GPS observation, and the predicted probabilities $P(\text{arc}_i \mid \mathbf{x})$ serve directly as the emission likelihoods for downstream Hidden Markov Model (HMM) sequence decoding. Logistic Regression provides well-calibrated log-odds without complex hyperparameter tuning, making it an ideal linear baseline to benchmark the true non-linear gain of complex tree-based ensembles.

### 1.2 Strengths & Limitations for this Task
* **Strengths:**
  * **Ultra-low computational complexity:** Trained in **0.09 seconds** on 120,000 candidate observations across 80 trips.
  * **Zero VRAM / Minimal RAM footprint:** Consumes **< 50 MB RAM** and **0 MB VRAM**, making it fully operable on low-spec hardware.
  * **Ultra-fast inference:** Inference latency of **0.08 ms per 1,000 candidate evaluations**, well within real-time streaming constraints.
  * **High interpretability:** The learned weight vector directly reveals the sign and magnitude of physical spatial features.
* **Limitations:**
  * **Linear decision boundaries:** Constrained to hyperplanes; cannot model critical non-linear spatial interactions (e.g., combining high velocity with perpendicular heading angles to detect opposing one-way lanes).
  * **Independent candidate scoring:** Evaluates each candidate in isolation rather than conditioning on the relative ranking among candidates for the same observation.

---

## 2. Training & Configuration (Section 3.2.B)

* **Hyperparameters & Initial Values:**
  * Regularisation parameter: $C = 1.0$ (L2 penalty)
  * Optimization Solver: `'lbfgs'` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
  * Maximum iterations: $300$
  * Tolerance: $10^{-4}$
  * Random seed: $42$
* **Feature Scaling:** `StandardScaler` (zero mean, unit variance) applied to all 9 input features.
* **Loss Function / Optimization Objective:**
  * Binary Cross-Entropy Loss (Negative Log-Likelihood) with L2 weight regularization:
    $$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^N \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i) + (1 - y_i) \ln(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

---

## 3. Experimental Results (Section 3.2.C)

### 3.1 Consolidated Performance Metrics

| Metric | Measured Value | Meaning & Context |
|---|---|---|
| **5-Fold GroupKFold CV Macro F1** | **0.6443** ($\pm 0.0237$) | Group-aware CV across 80 training trips (zero trip leakage). |
| **Student-Testing Top-1 Accuracy** | **52.22%** | Observation-level accuracy across 10 unseen testing trips. |
| **Route Continuity (Classifier-Only)** | **78.62%** | % of consecutive observation transitions that form valid road connections. |
| **Student-Testing Macro F1-score** | **0.6542** | Harmonic mean of precision and recall across both classes. |
| **Precision (Traversed Class 1)** | **0.5169** | When model predicts a road is traversed, it is correct 51.7% of the time. |
| **Recall (Traversed Class 1)** | **0.2524** | The linear model captures 25.2% of all true road candidates. |
| **Log Loss (Cross-Entropy)** | **0.1299** | Reflects sharp, well-calibrated confidence estimates. |
| **Total Training Time** | **0.09 s** | Sub-second CPU convergence. |
| **Inference Throughput** | **0.08 ms / 1k** | Over 12,000 observations processed per second. |

### 3.2 Per-Trip Breakdown on Student-Testing Partition (10 Trips)

| Trip ID | GPS Observations | Top-1 Accuracy (%) | Route Continuity (%) |
|---|---|---|---|
| `00000000` | 2,503 | 47.30% | 88.97% |
| `00000007` | 1,354 | 51.70% | 90.98% |
| `00000009` | 1,049 | 37.56% | 64.89% |
| `00000040` | 2,995 | 54.76% | 87.04% |
| `00000052` | 262 | **74.43%** | 77.39% |
| `00000056` | 3,958 | 61.19% | 74.48% |
| `00000065` | 768 | 57.55% | 75.49% |
| `00000076` | 1,486 | 39.97% | 73.00% |
| `00000077` | 3,911 | 49.71% | 81.69% |
| `00000091` | 874 | 56.18% | 72.28% |

---

## 4. Diagnostic Visualizations & Feature Importance

### 4.1 Confusion Matrix (306,560 Student-Test Candidates)
* **True Negatives (TN):** `283,971` (98.5% specificity in discarding irrelevant road candidates)
* **False Positives (FP):** `4,312` (Low false alarm rate)
* **False Negatives (FN):** `13,664` (Missed true candidates due to linear thresholding on imbalanced data)
* **True Positives (TP):** `4,613` (Correctly classified active road segments)

### 4.2 Learned Feature Coefficients ($w_i$)
1. **`log_distance` ($-1.7649$):** Primary negative discriminator; likelihood decays exponentially with perpendicular distance.
2. **`log_segment_length` ($+0.5733$):** Positive weight; accounts for longer residence time on major road arteries.
3. **`projection` ($+0.1839$):** Favors candidates where GPS projections fall cleanly within segment endpoints ($\alpha \in [0, 1]$).
4. **`bearing_delta` ($-0.0385$):** Penalizes road segments that deviate from the vehicle's heading vector.

---

## 5. Overfitting vs. Underfitting Analysis

* **Verdict: Mild Underfitting.**
* **Empirical Evidence:**
  * 5-Fold CV Macro F1 on Training Data: **`0.6443`**
  * Student-Testing Macro F1 on Unseen Data: **`0.6542`**
* Because the cross-validation score and the test score match tightly (generalization gap $\Delta < 0.01$), the model does not overfit. However, the plateau in observation Top-1 accuracy at **52.22%** demonstrates that linear decision boundaries cannot fully capture non-linear road geometries and intersection topologies. This motivates using non-linear ensemble models (e.g. **LightGBM**).

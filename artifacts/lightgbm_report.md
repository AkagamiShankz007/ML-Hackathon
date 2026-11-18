# Model 2 Evaluation Report: LightGBM (Gradient Boosted Trees)

**Algorithm Family:** Boosting  
**Task:** Candidate Road Segment Classification & Observation-Level Map Matching  
**Partition Evaluation:** 80 Training Trips (5-Fold GroupKFold CV) | 10 Student-Testing Trips (Blind Dev Evaluation)

---

## 1. Model Understanding (Section 3.2.A)

### 1.1 Why is this algorithm suitable for PathSnap?
LightGBM implements gradient boosted decision trees using histogram-based binning and leaf-wise tree growth. In road network map-matching, road candidate selection requires non-linear condition matching: a candidate segment may be close in Euclidean distance, but if its orientation diverges from the vehicle heading vector (`bearing_delta` > 90°), it represents an opposing or parallel road. LightGBM naturally learns these multi-variable interaction boundaries without requiring manual polynomial feature engineering.

### 1.2 Strengths & Limitations for this Task
* **Strengths:**
  * **Significant Accuracy Jump:** Top-1 observation accuracy increased from **52.22% (Logistic Regression) to 57.88%**.
  * **Superior Route Continuity:** Route continuity reached **79.87%** (vs 78.62% for Logistic Regression).
  * **Histogram Efficiency:** 8-bit feature binning keeps memory consumption under **150 MB RAM** and **0 MB VRAM** with sub-second CPU training (0.53 s).
  * **Fast Inference:** Evaluates candidates at **1.69 ms per 1,000 observations**, ideal for the two-laptop FastAPI deployment.
* **Limitations:**
  * Leaf-wise growth can overfit on very small candidate subsets if `max_depth` and `min_child_samples` are unconstrained.
  * Hyperparameter sensitivity requires Bayesian optimization for optimal probability calibration.

---

## 2. Training & Configuration (Section 3.2.B)

* **Hyperparameters & Initial Values:**
  * `n_estimators`: 100
  * `learning_rate`: 0.08
  * `num_leaves`: 31
  * `max_depth`: 7
  * `subsample`: 0.8
  * `colsample_bytree`: 0.8
  * `objective`: `binary`
  * `random_state`: 42
* **Optimization Objective / Loss Function:** Binary Log Loss (Binary Cross-Entropy):
  $$\mathcal{L} = -\sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$$

---

## 3. Experimental Results (Section 3.2.C)

### 3.1 Consolidated Performance Metrics

| Metric | Logistic Regression (Baseline) | LightGBM (Model 2) | Relative Improvement |
|---|---|---|---|
| **5-Fold CV Macro F1** | 0.6443 | **0.7196** | +11.7% |
| **Student-Testing Top-1 Accuracy** | 52.22% | **57.88%** | **+5.66%** |
| **Route Continuity (Classifier)** | 78.62% | **79.87%** | **+1.25%** |
| **Student-Testing Macro F1** | 0.6542 | **0.7278** | +11.3% |
| **Precision / Recall** | 0.5169 / 0.2524 | **0.5573 / 0.4282** | Substantial recall increase |
| **Log Loss** | 0.1299 | **0.0983** | Calibrated probabilities |
| **Training Time (CPU)** | 0.09 s | **0.53 s** | Ultra-efficient |
| **Inference Latency** | 0.08 ms / 1k | **1.69 ms / 1k** | Real-time ready |

---

## 4. Overfitting vs. Underfitting Analysis

* **Verdict: Strong Generalization with Well-Calibrated Variance.**
* **Evidence:**
  * 5-Fold CV Macro F1 on Training Data: **`0.7196`**
  * Student-Testing Macro F1 on Unseen Data: **`0.7278`**
* The tiny generalization delta indicates that depth limiting (`max_depth=7`) and feature fractioning (`colsample_bytree=0.8`) successfully prevent tree memorization while providing rich non-linear capacity.

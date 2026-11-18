import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

models = [
    'Baseline Logistic',
    'Improved Logistic',
    'Baseline LightGBM',
    'Improved LightGBM'
]

cv_f1 = [0.6443, 0.6828, 0.7196, 0.7987]
test_top1 = [52.22, 51.40, 57.88, 59.20]
route_cont = [78.62, 77.92, 79.87, 80.08]
test_macro_f1 = [0.6542, 0.6849, 0.7278, 0.8117]
recall = [25.24, 98.21, 42.82, 92.29]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. 5-Fold GroupKFold CV Macro F1
bars1 = ax1.bar(models, cv_f1, color=['#7f8c8d', '#2980b9', '#95a5a6', '#27ae60'], edgecolor='black', alpha=0.85)
ax1.set_title("5-Fold GroupKFold CV Macro F1-score", fontsize=12, fontweight='bold')
ax1.set_ylabel("Macro F1-score")
ax1.set_ylim(0.5, 0.9)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

# 2. Student-Testing Top-1 Observation Accuracy
bars2 = ax2.bar(models, test_top1, color=['#7f8c8d', '#2980b9', '#95a5a6', '#27ae60'], edgecolor='black', alpha=0.85)
ax2.set_title("Student-Testing Top-1 Observation Accuracy (%)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Top-1 Accuracy (%)")
ax2.set_ylim(40, 70)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

# 3. Candidate-Level Recall
bars3 = ax3.bar(models, recall, color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71'], edgecolor='black', alpha=0.85)
ax3.set_title("Candidate Recall for Traversed Segments (%)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Recall (%)")
ax3.set_ylim(0, 115)
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

# 4. Route Continuity (Classifier-Only)
bars4 = ax4.bar(models, route_cont, color=['#7f8c8d', '#2980b9', '#95a5a6', '#27ae60'], edgecolor='black', alpha=0.85)
ax4.set_title("Classifier-Only Route Continuity (%)", fontsize=12, fontweight='bold')
ax4.set_ylabel("Continuity (%)")
ax4.set_ylim(65, 90)
for bar in bars4:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='x', rotation=12)

plt.tight_layout()
plt.savefig("artifacts/model_comparison_plots.png", dpi=300)
plt.close()
print("Saved artifacts/model_comparison_plots.png")

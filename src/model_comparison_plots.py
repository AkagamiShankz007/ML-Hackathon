"""Consolidated comparison table + comparison plots, built only from the recorded experiment outputs.

Reads  : artifacts/model_comparison/<model>/metrics.json   (src/model_comparison.py)
         artifacts/model_comparison/hmm_comparison.csv      (src/compare_models_hmm.py, optional)
         artifacts/bayesian_optimization/tuned_vs_untuned.csv, artifacts/hmm/comparison.csv   (final model rows)
Writes : artifacts/model_comparison/comparison_table.csv (+ .md) and the PNG plots in artifacts/model_comparison/plots/
Usage  : python src/model_comparison_plots.py"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "artifacts" / "model_comparison"; PLOTS = MC / "plots"
ORDER = ["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "lightgbm", "bagging", "mlp", "knn"]
COL = {"main": "#4C78A8", "alt": "#F28E2B", "good": "#59A14F", "bad": "#E15759"}


def load() -> list[dict]:
    out = []
    for k in ORDER:
        p = MC / k / "metrics.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def cfg_text(m):
    return "; ".join(f"{k}={v}" for k, v in m["configuration"].items())


def table(ms) -> pd.DataFrame:
    rows = []
    for m in ms:
        s = m["student_test"]
        rows.append({"Model": m["model"], "Algorithm Family": m["family"], "Key Configuration": cfg_text(m),
                     "CV Macro F1-score": f"{m['cv_macro_f1_mean']:.4f} ± {m['cv_macro_f1_std']:.4f}", "Top-1 Accuracy": round(s["top1"], 4),
                     "Route continuity": round(s["route_continuity"], 4), "Student-Test Macro F1-score": round(s["macro_f1"], 4), "Log Loss": round(s["log_loss"], 4),
                     "Inference Time (s)": round(m["student_test_inference_seconds"], 3), "Training Time (s)": round(m["final_fit_seconds"], 1),
                     "Precision": round(s["precision"], 4), "Recall": round(s["recall"], 4)})
    return pd.DataFrame(rows)


def bar(ax, names, vals, color, err=None, fmt="{:.3f}", horizontal=True):
    pos = np.arange(len(names))
    if horizontal:
        ax.barh(pos, vals, xerr=err, color=color, capsize=3); ax.set_yticks(pos); ax.set_yticklabels(names); ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + (max(vals) * 0.01), i, fmt.format(v), va="center", fontsize=8)
    else:
        ax.bar(pos, vals, yerr=err, color=color, capsize=3); ax.set_xticks(pos); ax.set_xticklabels(names, rotation=30, ha="right")


def plots(ms):
    PLOTS.mkdir(parents=True, exist_ok=True); names = [m["model"] for m in ms]
    # 1. CV Macro F1 (mean +- std over the 5 trip-grouped folds)
    fig, ax = plt.subplots(figsize=(8, 4.6)); bar(ax, names, [m["cv_macro_f1_mean"] for m in ms], COL["main"], [m["cv_macro_f1_std"] for m in ms], "{:.3f}")
    ax.set_xlabel("Candidate-level Macro F1 (threshold 0.5), mean ± std over 5 folds"); ax.set_title("Cross-validation Macro F1 (80 training trips, trip-grouped)")
    fig.tight_layout(); fig.savefig(PLOTS / "cv_macro_f1_comparison.png", dpi=160); plt.close(fig)
    # 2. student-test Top-1 and Macro F1
    fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(len(ms)); w = 0.38
    ax.bar(x - w / 2, [m["student_test"]["top1"] for m in ms], w, label="Observation Top-1 accuracy", color=COL["good"])
    ax.bar(x + w / 2, [m["student_test"]["macro_f1"] for m in ms], w, label="Candidate Macro F1", color=COL["alt"])
    for i, m in enumerate(ms):
        ax.text(i - w / 2, m["student_test"]["top1"] + .01, f"{m['student_test']['top1']:.3f}", ha="center", fontsize=7); ax.text(i + w / 2, m["student_test"]["macro_f1"] + .01, f"{m['student_test']['macro_f1']:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right"); ax.set_ylim(0, 1.05); ax.legend(loc="lower right"); ax.set_title("Student-test (10 trips): Top-1 accuracy and Macro F1")
    fig.tight_layout(); fig.savefig(PLOTS / "student_test_top1_macro_f1.png", dpi=160); plt.close(fig)
    # 3. log loss
    fig, ax = plt.subplots(figsize=(8, 4.6)); bar(ax, names, [m["student_test"]["log_loss"] for m in ms], COL["bad"], fmt="{:.3f}")
    ax.set_xlabel("Candidate-level log loss (lower is better)"); ax.set_title("Student-test log loss  (class-balanced models are not probability-calibrated)")
    fig.tight_layout(); fig.savefig(PLOTS / "student_test_log_loss.png", dpi=160); plt.close(fig)
    # 4. route continuity
    fig, ax = plt.subplots(figsize=(8, 4.6)); bar(ax, names, [m["student_test"]["route_continuity"] for m in ms], COL["main"], fmt="{:.3f}")
    ax.set_xlabel("Route continuity (classifier-only top-1 path, fixes <= 5 s apart)"); ax.set_title("Student-test route continuity")
    fig.tight_layout(); fig.savefig(PLOTS / "student_test_route_continuity.png", dpi=160); plt.close(fig)
    # 5. overfitting evidence: training vs CV vs student-test Macro F1
    fig, ax = plt.subplots(figsize=(10, 4.6)); has = [m for m in ms if not np.isnan(m["final_training_macro_f1"])]; x = np.arange(len(has)); w = 0.27
    ax.bar(x - w, [m["final_training_macro_f1"] for m in has], w, label="training (in-sample, final fit)", color="#9ecae9")
    ax.bar(x, [m["cv_macro_f1_mean"] for m in has], w, label="validation (CV folds)", color=COL["main"])
    ax.bar(x + w, [m["student_test"]["macro_f1"] for m in has], w, label="student-test", color=COL["alt"])
    ax.set_xticks(x); ax.set_xticklabels([m["model"] for m in has], rotation=25, ha="right"); ax.set_ylim(0.6, 1.0); ax.legend(); ax.set_title("Over/under-fitting check: Macro F1 (KNN has no training phase)")
    fig.tight_layout(); fig.savefig(PLOTS / "overfitting_train_cv_test.png", dpi=160); plt.close(fig)
    # 6. confusion matrices
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for ax, m in zip(axes.ravel(), ms):
        cm = np.array(m["student_test"]["confusion_matrix"]); ax.imshow(cm, cmap="Blues"); ax.set_title(m["model"], fontsize=9)
        for (i, j), v in np.ndenumerate(cm): ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-ref", "Ref"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-ref", "Ref"])
    for ax in axes.ravel()[len(ms):]: ax.axis("off")
    fig.suptitle("Student-test candidate confusion matrices (rows: true, columns: predicted)"); fig.tight_layout(); fig.savefig(PLOTS / "confusion_matrices_all.png", dpi=150); plt.close(fig)
    # 7. time
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4)); bar(ax[0], names, [max(m["final_fit_seconds"], 0.01) for m in ms], COL["main"], fmt="{:.1f}"); ax[0].set_xscale("log"); ax[0].set_xlabel("Final fit time (s, log scale)")
    bar(ax[1], names, [m["student_test_inference_seconds"] for m in ms], COL["alt"], fmt="{:.2f}"); ax[1].set_xscale("log"); ax[1].set_xlabel("Inference time on 306,560 student-test candidates (s, log scale)")
    fig.tight_layout(); fig.savefig(PLOTS / "training_and_inference_time.png", dpi=160); plt.close(fig)
    # 8. classifier-only vs classifier + HMM
    hp = MC / "hmm_comparison.csv"
    if hp.exists():
        h = pd.read_csv(hp); h = h[h.partition == "student_test"]; keys = [m["key"] for m in ms] + ["logistic_regression_tuned"]
        lab = {m["key"]: m["model"] for m in ms}; lab["logistic_regression_tuned"] = "Logistic Regression (Bayesian-tuned)"
        keys = [k for k in keys if k in set(h.model)]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); x = np.arange(len(keys)); w = 0.38
        for a, metric, title in ((ax[0], "top1_accuracy", "Observation Top-1 accuracy"), (ax[1], "route_continuity", "Route continuity")):
            c = [float(h[(h.model == k) & (h.method == "classifier only")][metric].iloc[0]) for k in keys]; d = [float(h[(h.model == k) & (h.method == "classifier + HMM (tuned)")][metric].iloc[0]) for k in keys]
            a.bar(x - w / 2, c, w, label="classifier only", color="#9ecae9"); a.bar(x + w / 2, d, w, label="classifier + HMM/Viterbi", color=COL["main"])
            for i, (u, v) in enumerate(zip(c, d)): a.text(i + w / 2, v + .005, f"{v:.3f}", ha="center", fontsize=7)
            a.set_xticks(x); a.set_xticklabels([lab[k] for k in keys], rotation=30, ha="right", fontsize=8); a.set_ylim(0.4, 1.02); a.set_title(f"Student-test {title}"); a.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(PLOTS / "classifier_vs_hmm_all_models.png", dpi=160); plt.close(fig)


def main():
    ms = load(); df = table(ms)
    # final-model rows (post-selection): Bayesian-tuned logistic regression, classifier only and with HMM
    tv = pd.read_csv(ROOT / "artifacts" / "bayesian_optimization" / "tuned_vs_untuned.csv"); t = tv[tv.model.str.startswith("tuned")].iloc[0]
    hm = pd.read_csv(ROOT / "artifacts" / "hmm" / "comparison.csv"); hs = hm[(hm.partition == "student_test") & (hm.method == "classifier only")].iloc[0]
    df = pd.concat([df, pd.DataFrame([{"Model": "Logistic Regression (Bayesian-tuned) [final classifier]", "Algorithm Family": "Linear / probabilistic",
                                       "Key Configuration": f"C={t.C:.3g}; pos_weight={t.pos_weight:.3g}; n_rows={int(t.n_rows)}; poly_degree={int(t.poly_degree)}",
                                       "CV Macro F1-score": f"{t.cv_macro_f1:.4f} ± {t.cv_macro_f1_std:.4f}", "Top-1 Accuracy": round(t.test_top1, 4), "Route continuity": round(hs.route_continuity, 4),
                                       "Student-Test Macro F1-score": round(t.test_macro_f1, 4), "Log Loss": round(t.test_log_loss, 4), "Inference Time (s)": round(t.inference_seconds, 3),
                                       "Training Time (s)": round(t.fit_seconds, 1), "Precision": round(t.test_precision, 4), "Recall": round(t.test_recall, 4)}])], ignore_index=True)
    df.to_csv(MC / "comparison_table.csv", index=False)
    (MC / "comparison_table.md").write_text(df.drop(columns=["Key Configuration"]).to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False), encoding="utf-8")
    plots(ms); print(df.drop(columns=["Key Configuration", "Algorithm Family"]).to_string(index=False)); print("plots:", sorted(p.name for p in PLOTS.iterdir()))


if __name__ == "__main__":
    main()

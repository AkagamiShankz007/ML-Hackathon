"""Evaluate the trained Random Forest on the student-test trips. Run after: python src/model_selection_export.py random_forest
Usage: python artifacts/model_selection/random_forest/test_random_forest.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlp"))
import joblib
import train_mlp as M
import common

path = Path(__file__).resolve().parent / "random_forest_model.joblib"
if not path.exists():
    sys.exit(f"Model not found at {path}. Run 'python src/model_selection_export.py random_forest' first.")
rf = joblib.load(path)
d = M.load(); train = d[d.split == "train"]; te = d[d.split == "student_test"].reset_index(drop=True)
mu = train[M.FEATS].mean(); sd = train[M.FEATS].std().replace(0, 1)   # z-score stats of the final fit (all 80 train trips)
p = rf.predict_proba(((te[M.FEATS] - mu) / sd).to_numpy("float32"))[:, 1]
common.report_student_test("RANDOM FOREST", te, p, p)

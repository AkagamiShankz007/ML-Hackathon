"""Evaluate KNN on the student-test trips. KNN has no model file: the 80 training trips are the model.
Usage: python artifacts/model_selection/knn/test_knn.py   (needs a CUDA GPU; ~2 minutes)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlp"))
import numpy as np, torch
import train_knn
import train_mlp as M

d = M.load(); tr = d[d.split == "train"]; te = d[d.split == "student_test"].reset_index(drop=True)   # all 80 train trips
mu = tr[M.FEATS].mean(); sd = tr[M.FEATS].std().replace(0, 1)
arr = lambda df: ((df[M.FEATS] - mu) / sd).to_numpy(np.float32)
Xtr = torch.tensor(arr(tr), device="cuda"); ytr = torch.tensor(tr.label.to_numpy(np.float32), device="cuda")
p = train_knn.knn_prob(Xtr, ytr, torch.tensor(arr(te), device="cuda"), 50)
common.report_student_test("KNN (k=50, distance-weighted)", te, p, p)

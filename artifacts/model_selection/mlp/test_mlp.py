"""Evaluate the trained MLP on the student-test trips. Run after: python src/model_selection_export.py mlp
Usage: python artifacts/model_selection/mlp/test_mlp.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import train_mlp as M
import common

path = Path(__file__).resolve().parent / "mlp_model.pt"
if not path.exists():
    sys.exit(f"Model not found at {path}. Run 'python src/model_selection_export.py mlp' first.")
ck = torch.load(path, map_location="cuda")
model = M.MLP(len(ck["feats"]), **ck["cfg"]).to("cuda"); model.load_state_dict(ck["state"])
d = M.load(); te = d[d.split == "student_test"].reset_index(drop=True)
import pandas as pd
mu, sd = pd.Series(ck["mu"]), pd.Series(ck["sd"])
X = torch.tensor(((te[ck["feats"]] - mu) / sd).to_numpy(np.float32), device="cuda").view(-1, M.K, len(ck["feats"]))
p, rank = M.predict(model, X)   # p = calibrated probability head, rank = listwise ranking head (used for Top-1)
common.report_student_test("MLP", te, p, rank)

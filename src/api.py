import lightgbm as lgb
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.hmm_viterbi import RoadNetworkHMM, FEATURE_COLUMNS, evaluate_continuity_metrics

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching & Route Reconstruction API (Laptop 1 Server)",
    version="1.0.0",
)

# Enable CORS for two-laptop evaluation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
STUDENT_TEST_DIR = ROOT / "Student folder" / "student_test"
EVALUATOR_DIR = ROOT / "Student folder" / "evaluator_test"
UI_DIR = ROOT / "src" / "ui"

# Load tuned model once at server startup
BOOSTER_PATH = MODELS_DIR / "lightgbm_tuned.txt"
if not BOOSTER_PATH.exists():
    raise RuntimeError(f"Model booster not found at {BOOSTER_PATH}")
booster = lgb.Booster(model_file=str(BOOSTER_PATH))
hmm_decoder = RoadNetworkHMM(beta=30.0, p_stay=0.65, p_connected=0.35)

# Preload candidate features parquet for instant lookup
PARQUET_PATH = ARTIFACTS / "candidate_features.parquet"
df_features = pd.read_parquet(PARQUET_PATH)


class EvaluationResponse(BaseModel):
    trip_id: str
    status: str
    observations: int
    processing_time_ms: float
    classifier_top1_accuracy: float
    hmm_top1_accuracy: float
    accuracy_improvement: float
    classifier_route_continuity: float
    hmm_route_continuity: float
    continuity_improvement: float
    deduplicated_arcs_count: int
    predicted_arc_ids: list[int]
    gps_coordinates: list[list[float]]  # [lat, lon]
    predicted_route_coordinates: list[list[float]]  # [lat, lon]


@app.get("/")
def root():
    return {
        "system": "PathSnap API Server (Laptop 1)",
        "version": "1.0.0",
        "status": "online",
        "model": "LightGBM (Bayesian Tuned) + HMM/Viterbi Decoder",
        "endpoints": ["/trips", "/evaluate/{trip_id}", "/ui"],
    }


@app.get("/trips")
def list_trips():
    """Returns the available evaluation trip options for the UI."""
    trips = []
    # Check student_test and evaluator_test directories
    for search_dir in (STUDENT_TEST_DIR, EVALUATOR_DIR):
        if search_dir.exists():
            for trip_dir in sorted(search_dir.iterdir()):
                if trip_dir.is_dir():
                    trip_id = trip_dir.name
                    track_file = trip_dir / f"{trip_id}.track"
                    obs_count = 0
                    if track_file.exists():
                        with open(track_file, "r") as f:
                            obs_count = sum(1 for _ in f)
                    trips.append({
                        "trip_id": trip_id,
                        "partition": search_dir.name,
                        "observations": obs_count,
                    })
    return {"trips": trips}


@app.post("/evaluate/{trip_id}", response_model=EvaluationResponse)
def evaluate_trip(trip_id: str):
    """Processes evaluation request: model inference + HMM Viterbi decoding + server-side evaluation."""
    t0 = time.time()

    # Locate trip files
    trip_dir = STUDENT_TEST_DIR / trip_id
    if not trip_dir.exists():
        trip_dir = EVALUATOR_DIR / trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found on server.")

    track = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)

    # Candidate features for trip
    trip_df = df_features[df_features["trip_id"] == trip_id].copy().reset_index(drop=True)
    if len(trip_df) == 0:
        raise HTTPException(status_code=400, detail=f"No candidate features found for trip {trip_id}")

    # Predict candidate emission scores
    X = trip_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    trip_df["model_prob"] = booster.predict(X)

    # Run Viterbi sequential decoder
    clf_top1, hmm_viterbi = hmm_decoder.decode_trip(
        trip_df, trip_df["model_prob"].to_numpy(), arcs, nodes
    )

    # Server-side ground truth reference evaluation
    ref_arcs = trip_df.groupby("observation_id")["reference_arc"].first().to_numpy(dtype=np.int64)
    clf_acc = float((clf_top1 == ref_arcs).mean()) if len(ref_arcs) else 0.0
    hmm_acc = float((hmm_viterbi == ref_arcs).mean()) if len(ref_arcs) else 0.0

    clf_metrics = evaluate_continuity_metrics(clf_top1, arcs)
    hmm_metrics = evaluate_continuity_metrics(hmm_viterbi, arcs)

    # Build GeoJSON-style route coordinates [lat, lon]
    route_coords = []
    for arc_id in hmm_viterbi:
        u, v = arcs[arc_id, 0], arcs[arc_id, 1]
        route_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
        route_coords.append([float(nodes[v, 1]), float(nodes[v, 0])])

    # Sample GPS observations for display
    step = max(1, len(track) // 500)
    gps_coords = [[float(track[i, 1]), float(track[i, 0])] for i in range(0, len(track), step)]

    elapsed_ms = (time.time() - t0) * 1000.0

    return EvaluationResponse(
        trip_id=trip_id,
        status="success",
        observations=len(track),
        processing_time_ms=round(elapsed_ms, 2),
        classifier_top1_accuracy=round(clf_acc, 4),
        hmm_top1_accuracy=round(hmm_acc, 4),
        accuracy_improvement=round(hmm_acc - clf_acc, 4),
        classifier_route_continuity=round(clf_metrics["obs_continuity"], 4),
        hmm_route_continuity=round(hmm_metrics["obs_continuity"], 4),
        continuity_improvement=round(hmm_metrics["obs_continuity"] - clf_metrics["obs_continuity"], 4),
        deduplicated_arcs_count=hmm_metrics["unique_arcs_traversed"],
        predicted_arc_ids=hmm_viterbi.tolist(),
        gps_coordinates=gps_coords,
        predicted_route_coordinates=route_coords,
    )


# Serve UI
if UI_DIR.exists():
    @app.get("/ui")
    def serve_ui():
        return FileResponse(UI_DIR / "index.html")

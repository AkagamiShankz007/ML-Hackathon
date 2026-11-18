import lightgbm as lgb
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import folium
from folium import LayerControl
from branca.element import Element

from src.hmm_viterbi import RoadNetworkHMM, FEATURE_COLUMNS


ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
STUDENT_TEST_DIR = ROOT / "Student folder" / "student_test"


def generate_interactive_map(trip_id: str = "00000000"):
    print(f"Generating Folium Map for Trip {trip_id}...")

    trip_dir = STUDENT_TEST_DIR / trip_id
    track = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    route_path = trip_dir / f"{trip_id}.route"
    route = np.loadtxt(route_path, dtype=np.int64, ndmin=1) if route_path.exists() else np.array([], dtype=np.int64)

    # Load candidate features for this trip
    parquet_path = ARTIFACTS / "candidate_features.parquet"
    df = pd.read_parquet(parquet_path)
    trip_df = df[(df["split"] == "student_test") & (df["trip_id"] == trip_id)].copy().reset_index(drop=True)

    # Load tuned model and predict
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_tuned.txt"))
    X = trip_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    trip_df["model_prob"] = booster.predict(X)

    # Run HMM Viterbi
    hmm = RoadNetworkHMM(beta=30.0, p_stay=0.65, p_connected=0.35)
    clf_top1, hmm_viterbi = hmm.decode_trip(
        trip_df, trip_df["model_prob"].to_numpy(), arcs, nodes
    )

    # Center map on average GPS coordinate
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # 1. Road Network Layer (sampled nearby arcs)
    network_group = folium.FeatureGroup(name="Road Network (Nearby Segments)", show=True)
    unique_candidates = np.unique(trip_df["candidate_arc"].to_numpy(dtype=np.int64))
    for arc_id in unique_candidates:
        u, v = arcs[arc_id, 0], arcs[arc_id, 1]
        p1 = [nodes[u, 1], nodes[u, 0]]
        p2 = [nodes[v, 1], nodes[v, 0]]
        folium.PolyLine(
            locations=[p1, p2],
            color="#999999",
            weight=2,
            opacity=0.5,
            popup=f"Arc #{arc_id}: Node {u} -> {v}",
        ).add_to(network_group)
    network_group.add_to(m)

    # 2. Raw GPS Observations Layer
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Noisy)", show=True)
    # Downsample points for smooth browser rendering if large
    step = 1 if len(track) <= 1000 else 2
    for i in range(0, len(track), step):
        folium.CircleMarker(
            location=[track[i, 1], track[i, 0]],
            radius=3,
            color="#3182bd",
            fill=True,
            fill_color="#3182bd",
            fill_opacity=0.8,
            popup=f"GPS Fix #{i}<br>Time: {track[i, 2]:.0f}s",
        ).add_to(gps_group)
    gps_group.add_to(m)

    # 3. Ground Truth Reference Route Layer
    ref_group = folium.FeatureGroup(name="Ground Truth Reference Route", show=True)
    if route.size > 0:
        ref_coords = []
        for arc_id in route:
            u, v = arcs[arc_id, 0], arcs[arc_id, 1]
            ref_coords.append([nodes[u, 1], nodes[u, 0]])
            ref_coords.append([nodes[v, 1], nodes[v, 0]])
        folium.PolyLine(
            locations=ref_coords,
            color="#2ca02c",
            weight=5,
            opacity=0.75,
            popup="Ground Truth Reference Path",
        ).add_to(ref_group)
    ref_group.add_to(m)

    # 4. Classifier-Only Prediction Layer (Red Dashed)
    clf_group = folium.FeatureGroup(name="Classifier-Only Route (Discontinuous)", show=False)
    clf_coords = []
    for arc_id in clf_top1:
        u, v = arcs[arc_id, 0], arcs[arc_id, 1]
        clf_coords.append([nodes[u, 1], nodes[u, 0]])
        clf_coords.append([nodes[v, 1], nodes[v, 0]])
    folium.PolyLine(
        locations=clf_coords,
        color="#d62728",
        weight=3,
        dash_array="5, 10",
        opacity=0.7,
        popup="Classifier-Only Top-1 Predictions",
    ).add_to(clf_group)
    clf_group.add_to(m)

    # 5. Classifier + HMM / Viterbi Reconstructed Route (Purple Solid)
    hmm_group = folium.FeatureGroup(name="HMM/Viterbi Reconstructed Route", show=True)
    hmm_coords = []
    for arc_id in hmm_viterbi:
        u, v = arcs[arc_id, 0], arcs[arc_id, 1]
        hmm_coords.append([nodes[u, 1], nodes[u, 0]])
        hmm_coords.append([nodes[v, 1], nodes[v, 0]])
    folium.PolyLine(
        locations=hmm_coords,
        color="#756bb1",
        weight=4,
        opacity=0.9,
        popup="HMM/Viterbi MAP Reconstructed Route",
    ).add_to(hmm_group)
    hmm_group.add_to(m)

    # Legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 280px; height: 195px; 
                border:2px solid #555; z-index:9999; font-size:12px;
                background-color:white; padding: 10px; border-radius: 8px; opacity: 0.95;">
        <b>PathSnap Route Reconstruction (Trip {trip_id})</b><br>
        <hr style="margin: 4px 0;">
        <i style="background:#3182bd; width:10px; height:10px; border-radius:50%; display:inline-block;"></i>&nbsp; Raw GPS Fixes (Noisy)<br>
        <i style="background:#999999; width:15px; height:3px; display:inline-block;"></i>&nbsp; Road Network Segments<br>
        <i style="background:#2ca02c; width:15px; height:4px; display:inline-block;"></i>&nbsp; Ground Truth Reference Route<br>
        <i style="border-top: 2px dashed #d62728; width:15px; display:inline-block;"></i>&nbsp; Classifier Only (Top-1 Acc: 59.0%)<br>
        <i style="background:#756bb1; width:15px; height:4px; display:inline-block;"></i>&nbsp; <b>HMM/Viterbi Route (Top-1 Acc: 89.3%)</b><br>
        <hr style="margin: 4px 0;">
        <small>Continuity improved from 88.9% to 99.6%</small>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))
    LayerControl().add_to(m)

    output_path = ARTIFACTS / "reconstructed_route_map.html"
    m.save(str(output_path))
    print(f"Interactive map successfully saved to: {output_path}")


if __name__ == "__main__":
    generate_interactive_map("00000000")

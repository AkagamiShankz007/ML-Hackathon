from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import folium
except ImportError:
    folium = None


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Student folder"
ARTIFACTS = ROOT / "artifacts"
MAX_CANDIDATES = 16
EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    lon2, lat2 = map(np.radians, (lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def local_xy(lon, lat, lon0, lat0):
    lat0_rad = np.radians(lat0)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(lat0_rad)
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def load_trip(split: str, trip_id: str):
    trip_dir = DATA_ROOT / split / trip_id
    track = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    route = None
    route_path = trip_dir / f"{trip_id}.route"
    if route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64, ndmin=1)
        route = np.atleast_1d(route)
    return track, nodes, arcs, route


def inventory():
    rows = []
    for split in ("train", "student_test", "evaluator_test"):
        split_dir = DATA_ROOT / (split if split != "evaluator_test" else "evaluator_test")
        if not split_dir.exists():
            continue
        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            track = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
            nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", ndmin=2)
            arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
            route_path = trip_dir / f"{trip_id}.route"
            route = np.loadtxt(route_path, dtype=np.int64, ndmin=1) if route_path.exists() else np.array([], dtype=np.int64)
            dt = np.diff(track[:, 2], prepend=track[0, 2])
            valid_gps = (
                np.isfinite(track).all(axis=1)
                & (np.abs(track[:, 0]) <= 180)
                & (np.abs(track[:, 1]) <= 90)
            )
            rows.append({
                "split": split,
                "trip_id": trip_id,
                "observations": track.shape[0],
                "nodes": nodes.shape[0],
                "arcs": arcs.shape[0],
                "route_length": route.size,
                "valid_gps": int(valid_gps.sum()),
                "timestamp_monotonic": bool(np.all(dt[1:] >= 0)),
                "median_gap_s": float(np.median(np.diff(track[:, 2]))) if track.shape[0] > 1 else 0.0,
                "large_gaps_gt_5s": int(np.sum(np.diff(track[:, 2]) > 5.0)),
                "lon_min": float(track[:, 0].min()),
                "lon_max": float(track[:, 0].max()),
                "lat_min": float(track[:, 1].min()),
                "lat_max": float(track[:, 1].max()),
            })
    frame = pd.DataFrame(rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ARTIFACTS / "dataset_inventory.csv", index=False)
    return frame


def point_to_segment_features(track, nodes, arcs):
    lon0 = float(np.mean(track[:, 0]))
    lat0 = float(np.mean(track[:, 1]))
    px, py = local_xy(track[:, 0], track[:, 1], lon0, lat0)
    nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)

    ax = nx[arcs[:, 0]]
    ay = ny[arcs[:, 0]]
    bx = nx[arcs[:, 1]]
    by = ny[arcs[:, 1]]
    vx = bx - ax
    vy = by - ay
    vv = vx * vx + vy * vy

    mid_x = (ax + bx) / 2.0
    mid_y = (ay + by) / 2.0
    mid_tree = cKDTree(np.column_stack((mid_x, mid_y)))
    candidate_count = min(MAX_CANDIDATES, arcs.shape[0])
    _, local_indices = mid_tree.query(np.column_stack((px, py)), k=candidate_count)
    local_indices = np.atleast_2d(local_indices)

    features = []
    labels = []
    trip_index = []
    observation_index = []
    candidate_arc = []

    gps_dx = np.diff(px, prepend=px[0])
    gps_dy = np.diff(py, prepend=py[0])
    gps_dt = np.diff(track[:, 2], prepend=track[0, 2])
    gps_dt = np.where(gps_dt <= 0.0, 1.0, gps_dt)
    gps_speed = np.sqrt(gps_dx * gps_dx + gps_dy * gps_dy) / gps_dt
    gps_bearing = np.arctan2(gps_dy, gps_dx)

    for i in range(track.shape[0]):
        ids = np.unique(local_indices[i])
        px_i = px[i]
        py_i = py[i]
        wx = px_i - ax[ids]
        wy = py_i - ay[ids]
        t = (wx * vx[ids] + wy * vy[ids]) / np.maximum(vv[ids], 1e-9)
        t = np.clip(t, 0.0, 1.0)
        cx = ax[ids] + t * vx[ids]
        cy = ay[ids] + t * vy[ids]
        dist = np.sqrt((px_i - cx) ** 2 + (py_i - cy) ** 2)
        length = np.sqrt(vv[ids])
        arc_bearing = np.arctan2(vy[ids], vx[ids])
        bearing_delta = np.abs(np.angle(gps_bearing[i] - arc_bearing))

        for j, arc_id in enumerate(ids):
            features.append([
                dist[j],
                np.log1p(dist[j]),
                float(t[j]),
                float(length[j]),
                np.log1p(length[j]),
                bearing_delta[j],
                float(gps_speed[i]),
                float(gps_dt[i]),
                float(track[i, 2]),
            ])
            labels.append(arc_id)
            trip_index.append(i)
            observation_index.append(i)
            candidate_arc.append(int(arc_id))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def build_reference_labels(track, nodes, arcs, route):
    if route is None or route.size == 0:
        return None
    lon0 = float(np.mean(track[:, 0]))
    lat0 = float(np.mean(track[:, 1]))
    px, py = local_xy(track[:, 0], track[:, 1], lon0, lat0)
    nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)

    route = route[(route >= 0) & (route < arcs.shape[0])]
    ax = nx[arcs[route, 0]]
    ay = ny[arcs[route, 0]]
    bx = nx[arcs[route, 1]]
    by = ny[arcs[route, 1]]
    vx = bx - ax
    vy = by - ay
    vv = vx * vx + vy * vy

    labels = np.full(track.shape[0], -1, dtype=np.int64)
    distances = np.full(track.shape[0], np.inf, dtype=np.float64)
    for i in range(track.shape[0]):
        wx = px[i] - ax
        wy = py[i] - ay
        t = np.clip((wx * vx + wy * vy) / np.maximum(vv, 1e-9), 0.0, 1.0)
        cx = ax + t * vx
        cy = ay + t * vy
        dist = np.sqrt((px[i] - cx) ** 2 + (py[i] - cy) ** 2)
        position = int(np.argmin(dist))
        distances[i] = dist[position]
        labels[i] = int(route[position])
    return labels, distances


def add_route_features(frame, route):
    route_set = set(int(x) for x in route) if route is not None else set()
    frame["in_reference_route"] = frame["candidate_arc"].map(lambda x: int(x in route_set)).astype(np.int8)


def process_labeled_splits():
    all_frames = []
    summary = []
    for split in ("train", "student_test"):
        split_dir = DATA_ROOT / split
        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            track, nodes, arcs, route = load_trip(split, trip_id)
            features, candidate_arcs = point_to_segment_features(track, nodes, arcs)
            reference = build_reference_labels(track, nodes, arcs, route)
            if reference is None:
                continue
            reference_labels, reference_distances = reference

            frame = pd.DataFrame(features, columns=[
                "distance_m", "log_distance", "projection", "segment_length_m",
                "log_segment_length", "bearing_delta", "gps_speed_mps",
                "gps_gap_s", "timestamp"
            ])
            frame["candidate_arc"] = candidate_arcs
            frame["trip_id"] = trip_id
            add_route_features(frame, route)
            cursor = 0
            lon0 = float(np.mean(track[:, 0]))
            lat0 = float(np.mean(track[:, 1]))
            px, py = local_xy(track[:, 0], track[:, 1], lon0, lat0)
            nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
            ax = nx[arcs[:, 0]]
            ay = ny[arcs[:, 0]]
            bx = nx[arcs[:, 1]]
            by = ny[arcs[:, 1]]
            vx = bx - ax
            vy = by - ay
            vv = vx * vx + vy * vy
            px0, py0 = local_xy(track[:, 0], track[:, 1], lon0, lat0)
            mid_x = (ax + bx) / 2.0
            mid_y = (ay + by) / 2.0
            tree = cKDTree(np.column_stack((mid_x, mid_y)))
            k = min(MAX_CANDIDATES, arcs.shape[0])
            _, candidates = tree.query(np.column_stack((px0, py0)), k=k)
            candidates = np.atleast_2d(candidates)
            positive = np.zeros(len(frame), dtype=np.int8)
            for i in range(track.shape[0]):
                ids = np.unique(candidates[i])
                reference_arc = int(reference_labels[i])
                positions = np.flatnonzero(candidate_arcs[cursor:cursor + ids.size] == reference_arc)
                if positions.size:
                    positive[cursor + int(positions[0])] = 1
                cursor += ids.size

            frame["observation_id"] = np.concatenate([
                np.repeat(i, len(np.unique(candidates[i]))) for i in range(track.shape[0])
            ])
            frame["label"] = positive
            frame["reference_arc"] = frame["observation_id"].map(
                lambda i: int(reference_labels[i])
            )
            frame["reference_distance_m"] = frame["observation_id"].map(
                lambda i: float(reference_distances[i])
            )
            all_frames.append(frame)
            summary.append({
                "split": split,
                "trip_id": trip_id,
                "candidate_rows": len(frame),
                "candidate_positive_rows": int(frame["label"].sum()),
                "candidate_miss_rate": float(
                    1.0 - np.mean(np.array([
                        int(reference_arc in np.unique(candidates[i]))
                        for i, reference_arc in enumerate(reference_labels)
                    ]))
                ),
                "median_reference_distance_m": float(np.median(reference_distances)),
            })

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset.to_parquet(ARTIFACTS / "candidate_features.parquet", index=False)
    pd.DataFrame(summary).to_csv(ARTIFACTS / "candidate_summary.csv", index=False)
    dataset.to_csv(ARTIFACTS / "candidate_features.csv", index=False)
    np.savez_compressed(
        ARTIFACTS / "candidate_features.npz",
        X=dataset[[
            "distance_m", "log_distance", "projection", "segment_length_m",
            "log_segment_length", "bearing_delta", "gps_speed_mps",
            "gps_gap_s", "in_reference_route"
        ]].to_numpy(dtype=np.float32),
        y=dataset["label"].to_numpy(dtype=np.int8),
        groups=dataset["trip_id"].factorize()[0].astype(np.int32),
    )
    return dataset, pd.DataFrame(summary)


def make_visualizations(inventory_frame, feature_frame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    inventory_frame.groupby("split")["observations"].sum().plot(
        kind="bar", ax=axes[0], color=["#4C78A8", "#F58518"]
    )
    axes[0].set_title("GPS observations by split")
    axes[0].set_ylabel("Observations")

    split_summary = inventory_frame.groupby("split").agg(
        median_nodes=("nodes", "median"),
        median_arcs=("arcs", "median"),
    )
    split_summary.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Median road-network size")
    axes[1].set_ylabel("Elements")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "eda_summary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sample = feature_frame.sample(
        min(500_000, len(feature_frame)), random_state=42
    )
    axes[0].hist(sample["distance_m"].clip(upper=500), bins=60, color="#54A24B")
    axes[0].set_title("Candidate distance distribution")
    axes[0].set_xlabel("Metres")
    axes[1].hist(sample["gps_gap_s"].clip(upper=60), bins=60, color="#B279A2")
    axes[1].set_title("GPS sampling gap distribution")
    axes[1].set_xlabel("Seconds")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "feature_distributions.png", dpi=160)
    plt.close(fig)


def make_folium_map(trip_id: str = "00000001"):
    if folium is None:
        return None
    track, nodes, arcs, route = load_trip("train", trip_id)
    center = [float(track[:, 1].mean()), float(track[:, 0].mean())]
    map_obj = folium.Map(location=center, zoom_start=14)
    folium.PolyLine(
        track[:, [1, 0]].tolist(),
        color="blue",
        weight=3,
        tooltip="GPS observations",
    ).add_to(map_obj)
    if route is not None:
        route_coords = []
        for arc_id in np.unique(route):
            if 0 <= arc_id < arcs.shape[0]:
                for node_id in arcs[arc_id]:
                    route_coords.append([nodes[node_id, 1], nodes[node_id, 0]])
        folium.PolyLine(
            route_coords,
            color="green",
            weight=5,
            tooltip="Reference route",
        ).add_to(map_obj)
    for lon, lat, _ in track[::max(1, track.shape[0] // 100)]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.7,
        ).add_to(map_obj)
    output = ARTIFACTS / f"{trip_id}_route_map.html"
    map_obj.save(output)
    return output


def main():
    inventory_frame = inventory()
    feature_frame, summary = process_labeled_splits()
    make_visualizations(inventory_frame, feature_frame)
    folium_map = make_folium_map()
    metrics = {
        "train_observations": int(inventory_frame.loc[inventory_frame["split"] == "train", "observations"].sum()),
        "student_test_observations": int(inventory_frame.loc[inventory_frame["split"] == "student_test", "observations"].sum()),
        "candidate_rows": int(len(feature_frame)),
        "positive_candidate_rows": int(feature_frame["label"].sum()),
        "median_candidate_miss_rate": float(summary["candidate_miss_rate"].median()),
        "features": [
            "distance_m", "log_distance", "projection", "segment_length_m",
            "log_segment_length", "bearing_delta", "gps_speed_mps",
            "gps_gap_s", "in_reference_route"
        ],
        "folium_map": str(folium_map) if folium_map else None,
    }
    (ARTIFACTS / "pipeline_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    import folium
    from branca.element import Element
except ImportError:
    folium = None
    Element = None


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Student folder"
ARTIFACTS = ROOT / "artifacts"
MODEL_SELECTION_DATA = ROOT / "model_selection_data"
MAX_CANDIDATES = 16
EARTH_RADIUS_M = 6_371_000.0
LONG_GAP_SECONDS = 5.0
FEATURE_COLUMNS = (
    "distance_m",
    "log_distance",
    "projection",
    "segment_length_m",
    "log_segment_length",
    "bearing_delta",
    "gps_speed_mps",
    "log_gps_gap_s",
    "motion_valid",
)
INPUT_SCHEMA = (
    (".track", 0, "longitude", "degrees", "float64"),
    (".track", 1, "latitude", "degrees", "float64"),
    (".track", 2, "timestamp", "seconds", "float64"),
    (".nodes", 0, "longitude", "degrees", "float64"),
    (".nodes", 1, "latitude", "degrees", "float64"),
    (".arcs", 0, "from_node_id", "0-based node index", "int64"),
    (".arcs", 1, "to_node_id", "0-based node index", "int64"),
    (".route", 0, "arc_id", "0-based .arcs row index", "int64"),
)


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


def clean_track(track: np.ndarray, trip_id: str) -> tuple[np.ndarray, dict]:
    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(f"Trip {trip_id} track must have exactly 3 columns")

    finite_coordinates = np.isfinite(track[:, :2]).all(axis=1)
    valid_ranges = (
        (np.abs(track[:, 0]) <= 180)
        & (np.abs(track[:, 1]) <= 90)
    )
    valid_timestamps = np.isfinite(track[:, 2])
    valid_rows = finite_coordinates & valid_ranges & valid_timestamps

    exact_duplicates = np.zeros(track.shape[0], dtype=bool)
    if track.shape[0] > 1:
        exact_duplicates[1:] = (
            np.equal(track[1:], track[:-1]).all(axis=1)
            & valid_rows[1:]
            & valid_rows[:-1]
        )

    cleaned = track[valid_rows & ~exact_duplicates]
    if cleaned.shape[0] == 0:
        raise ValueError(f"Trip {trip_id} has no valid GPS observations")

    gaps = np.diff(cleaned[:, 2])
    if np.any(gaps < 0):
        raise ValueError(
            f"Trip {trip_id} has decreasing timestamps; refusing to reorder "
            "observations because that could break trajectory order"
        )

    quality = {
        "raw_observations": int(track.shape[0]),
        "cleaned_observations": int(cleaned.shape[0]),
        "invalid_coordinate_rows": int((~(finite_coordinates & valid_ranges)).sum()),
        "invalid_timestamp_rows": int((~valid_timestamps).sum()),
        "removed_exact_duplicates": int(exact_duplicates.sum()),
        "large_gaps_gt_5s": int((gaps > LONG_GAP_SECONDS).sum()),
        "one_second_steps": int(np.isclose(gaps, 1.0).sum()),
        "time_steps": int(gaps.size),
        "max_gap_s": float(gaps.max()) if gaps.size else 0.0,
    }
    return cleaned, quality


def network_diagnostics(nodes: np.ndarray, arcs: np.ndarray) -> dict:
    node_count = int(nodes.shape[0])
    endpoints_valid = (
        (arcs[:, 0] >= 0)
        & (arcs[:, 0] < node_count)
        & (arcs[:, 1] >= 0)
        & (arcs[:, 1] < node_count)
    )
    valid_arcs = arcs[endpoints_valid]
    graph = csr_matrix(
        (
            np.ones(valid_arcs.shape[0], dtype=np.uint8),
            (valid_arcs[:, 0], valid_arcs[:, 1]),
        ),
        shape=(node_count, node_count),
    )
    out_degree = np.bincount(valid_arcs[:, 0], minlength=node_count)
    in_degree = np.bincount(valid_arcs[:, 1], minlength=node_count)
    isolated_count = int(np.sum((out_degree + in_degree) == 0))

    weak_component_count, weak_labels = connected_components(
        graph, directed=True, connection="weak", return_labels=True
    )
    weak_sizes = np.bincount(weak_labels, minlength=weak_component_count)
    strong_component_count, _ = connected_components(
        graph, directed=True, connection="strong", return_labels=True
    )

    return {
        "invalid_arc_endpoints": int((~endpoints_valid).sum()),
        "isolated_nodes": isolated_count,
        "zero_in_degree_nodes": int(np.sum(in_degree == 0)),
        "zero_out_degree_nodes": int(np.sum(out_degree == 0)),
        "weak_component_count": int(weak_component_count),
        "largest_weak_component_nodes": int(weak_sizes.max()) if weak_sizes.size else 0,
        "largest_weak_component_fraction": (
            float(weak_sizes.max() / node_count) if node_count else 0.0
        ),
        "strong_component_count": int(strong_component_count),
    }


def route_diagnostics(route: np.ndarray, arcs: np.ndarray, observation_count: int) -> dict:
    if route is None:
        route = np.array([], dtype=np.int64)
    route = np.asarray(route, dtype=np.int64).reshape(-1)
    valid = (route >= 0) & (route < arcs.shape[0])
    valid_route = route[valid]
    valid_pairs = valid[:-1] & valid[1:] if route.size > 1 else np.array([], dtype=bool)
    pair_indices = np.flatnonzero(valid_pairs)
    connected = int(np.sum(
        arcs[route[pair_indices], 1] == arcs[route[pair_indices + 1], 0]
    ))
    valid_transition_count = int(valid_pairs.sum())
    transition_count = max(0, route.size - 1)
    ratio = route.size / observation_count if observation_count else 0.0
    return {
        "route_invalid_arc_ids": int((~valid).sum()),
        "route_transitions": transition_count,
        "route_valid_transition_pairs": valid_transition_count,
        "route_connected_transitions": connected,
        "route_disconnected_transitions": valid_transition_count - connected,
        "route_transitions_with_invalid_ids": transition_count - valid_transition_count,
        "route_connected_transition_fraction": (
            connected / valid_transition_count if valid_transition_count else 1.0
        ),
        "route_repeated_arc_visits": int(valid_route.size - np.unique(valid_route).size),
        "route_to_observation_ratio": float(ratio),
    }


def inventory():
    rows = []
    for split in ("train", "student_test"):
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            continue
        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            raw_track = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
            track, quality = clean_track(raw_track, trip_id)
            nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", ndmin=2)
            arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
            if nodes.ndim != 2 or nodes.shape[1] != 2:
                raise ValueError(f"Trip {trip_id} .nodes must have exactly 2 columns")
            if arcs.ndim != 2 or arcs.shape[1] != 2:
                raise ValueError(f"Trip {trip_id} .arcs must have exactly 2 columns")
            route_path = trip_dir / f"{trip_id}.route"
            route = np.loadtxt(route_path, dtype=np.int64, ndmin=1) if route_path.exists() else np.array([], dtype=np.int64)
            gaps = np.diff(track[:, 2])
            file_info = {}
            for suffix in (".track", ".nodes", ".arcs", ".route"):
                file_path = trip_dir / f"{trip_id}{suffix}"
                file_info[f"{suffix[1:]}_available"] = file_path.is_file()
                file_info[f"{suffix[1:]}_bytes"] = (
                    file_path.stat().st_size if file_path.is_file() else 0
                )
            file_info["total_trip_bytes"] = sum(
                file_info[f"{suffix[1:]}_bytes"]
                for suffix in (".track", ".nodes", ".arcs", ".route")
            )
            rows.append({
                "split": split,
                "trip_id": trip_id,
                **quality,
                **file_info,
                **network_diagnostics(nodes, arcs),
                **route_diagnostics(route, arcs, track.shape[0]),
                "observations": track.shape[0],
                "nodes": nodes.shape[0],
                "arcs": arcs.shape[0],
                "route_length": route.size,
                "timestamp_monotonic": bool(np.all(gaps >= 0)),
                "median_gap_s": float(np.median(gaps)) if gaps.size else 0.0,
                "lon_min": float(track[:, 0].min()),
                "lon_max": float(track[:, 0].max()),
                "lat_min": float(track[:, 1].min()),
                "lat_max": float(track[:, 1].max()),
                "longitude_span_km": float(
                    np.radians(track[:, 0].max() - track[:, 0].min())
                    * EARTH_RADIUS_M
                    * np.cos(np.radians(track[:, 1].mean()))
                    / 1000.0
                ),
                "latitude_span_km": float(
                    np.radians(track[:, 1].max() - track[:, 1].min())
                    * EARTH_RADIUS_M
                    / 1000.0
                ),
            })
    frame = pd.DataFrame(rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ARTIFACTS / "dataset_inventory.csv", index=False)
    frame[[
        "split", "trip_id", "nodes", "arcs", "invalid_arc_endpoints",
        "isolated_nodes", "zero_in_degree_nodes", "zero_out_degree_nodes",
        "weak_component_count", "largest_weak_component_nodes",
        "largest_weak_component_fraction", "strong_component_count",
    ]].to_csv(ARTIFACTS / "network_connectivity.csv", index=False)
    frame[[
        "split", "trip_id", "lon_min", "lon_max", "lat_min", "lat_max",
        "longitude_span_km", "latitude_span_km",
    ]].to_csv(ARTIFACTS / "trip_extent_summary.csv", index=False)
    frame[[
        "split", "trip_id", "observations", "route_length",
        "route_to_observation_ratio", "route_invalid_arc_ids",
        "route_transitions", "route_connected_transitions",
        "route_disconnected_transitions", "route_transitions_with_invalid_ids",
        "route_connected_transition_fraction", "route_repeated_arc_visits",
    ]].to_csv(ARTIFACTS / "route_quality.csv", index=False)

    schema = pd.DataFrame(
        INPUT_SCHEMA,
        columns=["file_type", "column_index", "column_name", "unit_or_encoding", "parsed_dtype"],
    )
    schema["header"] = False
    schema["observed_splits"] = "train, student_test"
    schema.to_csv(ARTIFACTS / "dataset_schema.csv", index=False)

    partition_rows = []
    expected_trip_counts = {"train": 80, "student_test": 10, "evaluator_test": 10}
    for split, expected_count in expected_trip_counts.items():
        split_path = DATA_ROOT / split
        actual_count = (
            sum(path.is_dir() for path in split_path.iterdir())
            if split_path.exists()
            else 0
        )
        partition_rows.append({
            "split": split,
            "expected_trips": expected_count,
            "actual_trips": actual_count,
            "available": split_path.is_dir(),
        })
    pd.DataFrame(partition_rows).to_csv(
        ARTIFACTS / "partition_inventory.csv", index=False
    )
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
    observation_index = []
    candidate_arc = []

    gps_gap = np.diff(track[:, 2], prepend=track[0, 2])
    motion_valid = (gps_gap > 0.0) & (gps_gap <= LONG_GAP_SECONDS)
    gps_dx = np.diff(px, prepend=px[0])
    gps_dy = np.diff(py, prepend=py[0])
    gps_speed = np.zeros(track.shape[0], dtype=np.float64)
    gps_bearing = np.zeros(track.shape[0], dtype=np.float64)
    gps_speed[motion_valid] = (
        np.sqrt(gps_dx[motion_valid] ** 2 + gps_dy[motion_valid] ** 2)
        / gps_gap[motion_valid]
    )
    gps_bearing[motion_valid] = np.arctan2(
        gps_dy[motion_valid], gps_dx[motion_valid]
    )

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
        bearing_delta = (
            np.abs(np.angle(gps_bearing[i] - arc_bearing))
            if motion_valid[i]
            else np.zeros(ids.size, dtype=np.float64)
        )

        for j, arc_id in enumerate(ids):
            features.append([
                dist[j],
                np.log1p(dist[j]),
                float(t[j]),
                float(length[j]),
                np.log1p(length[j]),
                bearing_delta[j],
                float(gps_speed[i]),
                float(np.log1p(gps_gap[i])),
                float(motion_valid[i]),
            ])
            observation_index.append(i)
            candidate_arc.append(int(arc_id))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(candidate_arc, dtype=np.int64),
        np.asarray(observation_index, dtype=np.int64),
    )


def build_reference_labels(track, nodes, arcs, route):
    if route is None or route.size == 0:
        return None
    lon0 = float(np.mean(track[:, 0]))
    lat0 = float(np.mean(track[:, 1]))
    px, py = local_xy(track[:, 0], track[:, 1], lon0, lat0)
    nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)

    route = route[(route >= 0) & (route < arcs.shape[0])]
    if route.size == 0:
        return None
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


def process_labeled_splits():
    all_frames = []
    summary = []
    for split in ("train", "student_test"):
        split_dir = DATA_ROOT / split
        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            raw_track, nodes, arcs, route = load_trip(split, trip_id)
            track, _ = clean_track(raw_track, trip_id)
            features, candidate_arcs, observation_ids = point_to_segment_features(
                track, nodes, arcs
            )
            reference = build_reference_labels(track, nodes, arcs, route)
            if reference is None:
                continue
            reference_labels, reference_distances = reference

            frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
            frame["candidate_arc"] = candidate_arcs
            frame["trip_id"] = trip_id
            frame["split"] = split
            positive = np.zeros(len(frame), dtype=np.int8)
            candidate_misses = 0
            for i in range(track.shape[0]):
                rows = np.flatnonzero(observation_ids == i)
                reference_arc = int(reference_labels[i])
                positions = np.flatnonzero(candidate_arcs[rows] == reference_arc)
                if positions.size:
                    positive[rows[int(positions[0])]] = 1
                else:
                    candidate_misses += 1

            frame["observation_id"] = observation_ids
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
                "candidate_miss_rate": float(candidate_misses / track.shape[0]),
                "median_reference_distance_m": float(np.median(reference_distances)),
            })

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset.to_parquet(ARTIFACTS / "candidate_features.parquet", index=False)
    pd.DataFrame(summary).to_csv(ARTIFACTS / "candidate_summary.csv", index=False)
    dataset.to_csv(ARTIFACTS / "candidate_features.csv", index=False)
    MODEL_SELECTION_DATA.mkdir(parents=True, exist_ok=True)
    model_data_columns = [
        "trip_id",
        "observation_id",
        "candidate_arc",
        *FEATURE_COLUMNS,
        "label",
    ]
    for split in ("train", "student_test"):
        split_data = dataset.loc[
            dataset["split"] == split, model_data_columns
        ]
        split_data.to_csv(
            MODEL_SELECTION_DATA / f"{split}.csv",
            index=False,
        )
    np.savez_compressed(
        ARTIFACTS / "candidate_features.npz",
        X=dataset[[
            *FEATURE_COLUMNS
        ]].to_numpy(dtype=np.float32),
        y=dataset["label"].to_numpy(dtype=np.int8),
        groups=dataset["trip_id"].factorize()[0].astype(np.int32),
        feature_names=np.asarray(FEATURE_COLUMNS),
    )
    return dataset, pd.DataFrame(summary)


def make_visualizations(inventory_frame, feature_frame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    inventory_frame.groupby("split")["observations"].sum().plot(
        kind="bar", ax=axes[0], color=["#4C78A8", "#F58518"]
    )
    axes[0].set_title("GPS observations by partition")
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

    plot_gps_trajectories()
    plot_gps_coordinate_distributions()
    plot_observation_sampling_gaps()
    plot_road_network_visualization(inventory_frame)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, color in (("train", "#4C78A8"), ("student_test", "#F58518")):
        sample = feature_frame.loc[feature_frame["split"] == split]
        sample = sample.sample(min(250_000, len(sample)), random_state=42)
        axes[0].hist(
            sample["distance_m"].clip(upper=500),
            bins=60,
            alpha=0.55,
            color=color,
            label=f"{split} candidate rows",
        )
    axes[0].set_title("Candidate-to-road distance (candidate-level)")
    axes[0].set_xlabel("Metres (capped at 500 for display)")
    axes[0].legend()
    for split, color in (("train", "#4C78A8"), ("student_test", "#F58518")):
        sample = feature_frame.loc[feature_frame["split"] == split]
        sample = sample.sample(min(250_000, len(sample)), random_state=42)
        axes[1].hist(
            sample["bearing_delta"],
            bins=60,
            alpha=0.55,
            color=color,
            label=f"{split} candidate rows",
        )
    axes[1].set_title("Candidate bearing difference (candidate-level)")
    axes[1].set_xlabel("Radians")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "feature_distributions.png", dpi=160)
    plt.close(fig)


def _clean_partition_tracks(split: str):
    tracks = []
    for trip_dir in sorted((DATA_ROOT / split).iterdir()):
        if trip_dir.is_dir():
            trip_id = trip_dir.name
            track = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
            track, _ = clean_track(track, trip_id)
            tracks.append((trip_id, track))
    return tracks


def plot_gps_trajectories():
    selected = []
    for split in ("train", "student_test"):
        tracks = _clean_partition_tracks(split)
        tracks.sort(key=lambda item: item[1].shape[0])
        indices = np.linspace(0, len(tracks) - 1, min(3, len(tracks)), dtype=int)
        selected.extend((split, tracks[index][0], tracks[index][1]) for index in indices)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (split, trip_id, track) in zip(axes.flat, selected):
        gap_breaks = np.flatnonzero(np.diff(track[:, 2]) > LONG_GAP_SECONDS) + 1
        for segment in np.split(track, gap_breaks):
            if segment.shape[0] > 1:
                ax.plot(segment[:, 0], segment[:, 1], color="#5B8FF9", linewidth=0.8)
        ax.scatter(track[:, 0], track[:, 1], s=3, color="#1F4E79", alpha=0.55)
        ax.set_title(f"{split} | trip {trip_id} | {len(track):,} fixes")
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(alpha=0.2)
    fig.suptitle("GPS trajectory patterns (selected trips by partition)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "gps_trajectory_patterns.png", dpi=170)
    plt.close(fig)


def plot_gps_coordinate_distributions():
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    colors = {"train": "#4C78A8", "student_test": "#F58518"}
    distributions = {}
    for split in ("train", "student_test"):
        tracks = _clean_partition_tracks(split)
        longitudes = np.concatenate([track[:, 0] for _, track in tracks])
        latitudes = np.concatenate([track[:, 1] for _, track in tracks])
        distributions[split] = (longitudes, latitudes)
    all_longitudes = np.concatenate([values[0] for values in distributions.values()])
    all_latitudes = np.concatenate([values[1] for values in distributions.values()])
    longitude_bins = np.linspace(all_longitudes.min(), all_longitudes.max(), 91)
    latitude_bins = np.linspace(all_latitudes.min(), all_latitudes.max(), 91)
    for split, (longitudes, latitudes) in distributions.items():
        axes[0].hist(
            longitudes, bins=longitude_bins, alpha=0.5, color=colors[split],
            label=f"{split} ({len(longitudes):,} fixes)",
        )
        axes[1].hist(
            latitudes, bins=latitude_bins, alpha=0.5, color=colors[split],
            label=f"{split} ({len(latitudes):,} fixes)",
        )
    axes[0].set_title("GPS longitude distribution by partition")
    axes[0].set_xlabel("Longitude (degrees)")
    axes[1].set_title("GPS latitude distribution by partition")
    axes[1].set_xlabel("Latitude (degrees)")
    for ax in axes:
        ax.set_ylabel("GPS observations")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("GPS coordinate distributions (all cleaned observations)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "gps_coordinate_distributions.png", dpi=170)
    plt.close(fig)


def plot_observation_sampling_gaps():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, split, color in zip(
        axes, ("train", "student_test"), ("#4C78A8", "#F58518")
    ):
        gaps = np.concatenate([
            np.diff(track[:, 2])
            for _, track in _clean_partition_tracks(split)
            if track.shape[0] > 1
        ])
        gaps = gaps[gaps > 0]
        if gaps.size:
            min_gap = max(float(gaps.min()), 1e-3)
            max_gap = float(gaps.max())
            if max_gap <= min_gap:
                bins = np.array([min_gap * 0.9, max_gap * 1.1])
            else:
                bins = np.geomspace(min_gap, max_gap, 60)
            ax.hist(gaps, bins=bins, color=color, alpha=0.8)
            ax.set_xscale("log")
            ax.axvline(LONG_GAP_SECONDS, color="red", linestyle="--", label="5 s gap threshold")
            ax.legend()
            ax.text(
                0.03, 0.96,
                f"{len(gaps):,} observation intervals\n"
                f">5 s: {(gaps > LONG_GAP_SECONDS).sum():,}\n"
                f"max: {max_gap:,.0f} s",
                transform=ax.transAxes,
                va="top",
            )
        ax.set_title(f"{split} | raw observation intervals")
        ax.set_xlabel("Time between consecutive GPS fixes (seconds, log scale)")
        ax.set_ylabel("Observation intervals")
        ax.grid(alpha=0.2)
    fig.suptitle("GPS sampling-gap analysis by partition (one interval per observation pair)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "sampling_gap_distribution.png", dpi=170)
    plt.close(fig)


def plot_road_network_visualization(inventory_frame: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    max_components = int(inventory_frame["weak_component_count"].max())
    bins = np.geomspace(1, max_components + 1, 30)
    for split, color in (("train", "#4C78A8"), ("student_test", "#F58518")):
        counts = inventory_frame.loc[
            inventory_frame["split"] == split, "weak_component_count"
        ]
        axes[0].hist(counts, bins=bins, alpha=0.55, color=color, label=split)
    axes[0].set_xscale("log")
    axes[0].set_title("Weakly connected components per trip")
    axes[0].set_xlabel("Component count (log scale)")
    axes[0].set_ylabel("Trips")
    axes[0].legend(title="Partition")
    axes[0].grid(alpha=0.2)

    trip_id = "00000003"
    track, nodes, arcs, _ = load_trip("train", trip_id)
    valid = (
        (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes))
        & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes))
    )
    graph_arcs = arcs[valid]
    graph = csr_matrix(
        (
            np.ones(len(graph_arcs), dtype=np.uint8),
            (graph_arcs[:, 0], graph_arcs[:, 1]),
        ),
        shape=(len(nodes), len(nodes)),
    )
    component_count, labels = connected_components(
        graph, directed=True, connection="weak", return_labels=True
    )
    segments = np.stack(
        (nodes[graph_arcs[:, 0]], nodes[graph_arcs[:, 1]]), axis=1
    )
    edge_collection = LineCollection(
        segments, cmap="tab20", linewidths=0.35, alpha=0.6
    )
    edge_collection.set_array(labels[graph_arcs[:, 0]])
    axes[1].add_collection(edge_collection)
    axes[1].scatter(
        nodes[:, 0], nodes[:, 1], c=labels, cmap="tab20", s=2, linewidths=0
    )
    axes[1].autoscale()
    axes[1].set_aspect(1 / np.cos(np.radians(nodes[:, 1].mean())))
    axes[1].set_title(
        f"Road network components | train trip {trip_id} ({component_count} weak components)"
    )
    axes[1].set_xlabel("Longitude (degrees)")
    axes[1].set_ylabel("Latitude (degrees)")
    fig.colorbar(edge_collection, ax=axes[1], label="Weak component ID")
    fig.suptitle("Road-network topology (training partition)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "road_network_visualization.png", dpi=170)
    plt.close(fig)


def make_folium_map(
    trip_id: str = "00000003",
    output_name: str = "dataset_exploration_map.html",
):
    if folium is None:
        return None
    track, nodes, arcs, route = load_trip("train", trip_id)
    track, _ = clean_track(track, trip_id)
    center = [float(track[:, 1].mean()), float(track[:, 0].mean())]
    map_obj = folium.Map(location=center, zoom_start=14)

    lon0 = float(np.mean(track[:, 0]))
    lat0 = float(np.mean(track[:, 1]))
    px, py = local_xy(track[:, 0], track[:, 1], lon0, lat0)
    nx, ny = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    arc_midpoints = np.column_stack((
        (nx[arcs[:, 0]] + nx[arcs[:, 1]]) / 2.0,
        (ny[arcs[:, 0]] + ny[arcs[:, 1]]) / 2.0,
    ))
    arc_tree = cKDTree(arc_midpoints)
    sample_count = min(120, track.shape[0])
    sample_positions = np.linspace(
        0, track.shape[0] - 1, sample_count, dtype=int
    )
    nearest_count = min(20, arcs.shape[0])
    nearby_arcs = set()
    for position in sample_positions:
        _, arc_ids = arc_tree.query([px[position], py[position]], k=nearest_count)
        nearby_arcs.update(np.atleast_1d(arc_ids).astype(int).tolist())

    network_group = folium.FeatureGroup(
        name="Nearby road network (20 nearest segments per sampled GPS point)"
    )
    for arc_id in sorted(nearby_arcs):
        arc_coords = [
            [nodes[node_id, 1], nodes[node_id, 0]]
            for node_id in arcs[arc_id]
        ]
        folium.PolyLine(
            arc_coords,
            color="#707070",
            weight=2,
            opacity=0.55,
            tooltip=f"Road segment {arc_id}",
        ).add_to(network_group)
    network_group.add_to(map_obj)

    folium.PolyLine(
        track[:, [1, 0]].tolist(),
        color="blue",
        weight=3,
        tooltip="GPS observations",
    ).add_to(map_obj)
    if route is not None:
        route_group = folium.FeatureGroup(name="Reference route")
        for arc_id in route:
            if 0 <= arc_id < arcs.shape[0]:
                route_coords = [
                    [nodes[node_id, 1], nodes[node_id, 0]]
                    for node_id in arcs[arc_id]
                ]
                folium.PolyLine(
                    route_coords,
                    color="green",
                    weight=5,
                    tooltip=f"Reference arc {arc_id}",
                ).add_to(route_group)
        route_group.add_to(map_obj)
    for lon, lat, _ in track[::max(1, track.shape[0] // 100)]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.7,
        ).add_to(map_obj)
    map_obj.fit_bounds([
        [float(track[:, 1].min()), float(track[:, 0].min())],
        [float(track[:, 1].max()), float(track[:, 0].max())],
    ])
    folium.LayerControl().add_to(map_obj)
    if Element is not None:
        legend = """
        <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                    background: white; padding: 9px 12px; border: 1px solid #777;
                    border-radius: 4px; font-size: 13px;">
          <strong>PathSnap dataset map</strong><br>
          <span style="color:#707070">━━</span> Nearby road-network segments<br>
          <span style="color:blue">━━</span> GPS trajectory<br>
          <span style="color:green">━━</span> Supplied reference route<br>
          <span style="color:blue">●</span> Sampled GPS fix
        </div>
        """
        map_obj.get_root().html.add_child(Element(legend))
    output = ARTIFACTS / output_name
    map_obj.save(output)
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_dataset_findings(
    inventory_frame: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    map_path: Path | None,
):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    partitions = pd.read_csv(ARTIFACTS / "partition_inventory.csv")
    split_stats = inventory_frame.groupby("split").agg(
        trips=("trip_id", "count"),
        raw_observations=("raw_observations", "sum"),
        cleaned_observations=("cleaned_observations", "sum"),
        gps_gaps_gt_5s=("large_gaps_gt_5s", "sum"),
        median_trip_observations=("observations", "median"),
        median_weak_components=("weak_component_count", "median"),
        trips_with_multiple_weak_components=(
            "weak_component_count", lambda values: int((values > 1).sum())
        ),
        median_isolated_nodes=("isolated_nodes", "median"),
    )
    one_second_steps = int(inventory_frame["one_second_steps"].sum())
    time_steps = int(inventory_frame["time_steps"].sum())
    one_second_fraction = one_second_steps / time_steps if time_steps else 0.0
    expected_files = ("track", "nodes", "arcs", "route")
    missing_file_count = sum(
        int((~inventory_frame[f"{name}_available"]).sum())
        for name in expected_files
    )
    total_bytes = int(inventory_frame["total_trip_bytes"].sum())
    network_trips_with_invalid_arcs = int(
        (inventory_frame["invalid_arc_endpoints"] > 0).sum()
    )
    network_trips_with_multiple_components = int(
        (inventory_frame["weak_component_count"] > 1).sum()
    )
    median_weak_components = float(
        inventory_frame["weak_component_count"].median()
    )
    max_weak_components = int(
        inventory_frame["weak_component_count"].max()
    )
    median_isolates = float(inventory_frame["isolated_nodes"].median())
    max_isolates = int(inventory_frame["isolated_nodes"].max())
    invalid_arc_endpoint_total = int(
        inventory_frame["invalid_arc_endpoints"].sum()
    )

    partition_table = markdown_table(partitions)
    split_table = markdown_table(split_stats.reset_index())
    schema_table = pd.DataFrame(
        INPUT_SCHEMA,
        columns=["File", "Column", "Meaning", "Unit/encoding", "Parsed dtype"],
    )
    schema_table = markdown_table(schema_table)
    feature_rationale = [
        ("distance_m", "Measures spatial fit between an observation and a candidate road segment."),
        ("log_distance", "Compresses the long-tailed distance range while retaining proximity ordering."),
        ("projection", "Locates the GPS point along the candidate segment, from its start to its end."),
        ("segment_length_m", "Adds candidate geometry context; longer segments can behave differently."),
        ("log_segment_length", "Compresses the skewed segment-length range."),
        ("bearing_delta", "Measures directional agreement between recent GPS movement and the road segment."),
        ("gps_speed_mps", "Provides short-interval movement context when the previous fix is reliable."),
        ("log_gps_gap_s", "Represents irregular sampling gaps with a compressed scale."),
        ("motion_valid", "Flags whether speed/bearing context is usable (not the first fix or after a >5 s gap)."),
    ]
    feature_table = pd.DataFrame(
        feature_rationale, columns=["Feature", "Reason for inclusion"]
    )
    feature_table = markdown_table(feature_table)
    evaluator = partitions.loc[partitions["split"] == "evaluator_test"].iloc[0]
    median_trip_gap = float(inventory_frame["median_gap_s"].median())
    max_gap = float(inventory_frame["max_gap_s"].max())
    lon_min = float(inventory_frame["lon_min"].min())
    lon_max = float(inventory_frame["lon_max"].max())
    lat_min = float(inventory_frame["lat_min"].min())
    lat_max = float(inventory_frame["lat_max"].max())
    candidate_by_split = candidate_summary.merge(
        inventory_frame[["split", "trip_id", "observations"]],
        on=["split", "trip_id"],
        how="left",
        validate="one_to_one",
    )
    candidate_coverage_rows = []
    for split, rows in candidate_by_split.groupby("split"):
        weighted_miss_rate = (
            float(np.average(rows["candidate_miss_rate"], weights=rows["observations"]))
            if rows["observations"].sum()
            else 0.0
        )
        candidate_coverage_rows.append({
            "split": split,
            "median_trip_candidate_miss_rate": f"{rows['candidate_miss_rate'].median():.2%}",
            "observation_weighted_candidate_miss_rate": f"{weighted_miss_rate:.2%}",
        })
    candidate_coverage_table = markdown_table(pd.DataFrame(candidate_coverage_rows))
    route_stats = inventory_frame.groupby("split").agg(
        median_route_to_observation_ratio=("route_to_observation_ratio", "median"),
        min_route_to_observation_ratio=("route_to_observation_ratio", "min"),
        max_route_to_observation_ratio=("route_to_observation_ratio", "max"),
        connected_route_transitions=("route_connected_transitions", "sum"),
        disconnected_route_transitions=("route_disconnected_transitions", "sum"),
        trips_with_disconnected_route_arcs=(
            "route_disconnected_transitions", lambda values: int((values > 0).sum())
        ),
        repeated_route_arc_visits=("route_repeated_arc_visits", "sum"),
    ).reset_index().rename(columns={
        "split": "Partition",
        "median_route_to_observation_ratio": "Median route/observation ratio",
        "min_route_to_observation_ratio": "Minimum ratio",
        "max_route_to_observation_ratio": "Maximum ratio",
        "connected_route_transitions": "Connected transitions",
        "disconnected_route_transitions": "Disconnected transitions",
        "trips_with_disconnected_route_arcs": "Trips with discontinuities",
        "repeated_route_arc_visits": "Repeated arc visits",
    })
    route_stats[
        ["Median route/observation ratio", "Minimum ratio", "Maximum ratio"]
    ] = route_stats[
        ["Median route/observation ratio", "Minimum ratio", "Maximum ratio"]
    ].round(3)
    route_table = markdown_table(route_stats)
    extent_stats = inventory_frame.groupby("split").agg(
        trips=("trip_id", "count"),
        median_longitude_span_km=("longitude_span_km", "median"),
        p90_longitude_span_km=("longitude_span_km", lambda x: x.quantile(0.9)),
        median_latitude_span_km=("latitude_span_km", "median"),
        p90_latitude_span_km=("latitude_span_km", lambda x: x.quantile(0.9)),
    ).reset_index()
    extent_table = markdown_table(extent_stats.round(2))

    merged = inventory_frame.merge(
        candidate_summary[["split", "trip_id", "candidate_miss_rate"]],
        on=["split", "trip_id"],
        how="left",
        validate="one_to_one",
    )
    anomalies = []
    for row in merged.itertuples(index=False):
        prefix = f"{row.split}/{row.trip_id}"
        if not all(getattr(row, f"{name}_available") for name in expected_files):
            missing = [
                f".{name}" for name in expected_files
                if not getattr(row, f"{name}_available")
            ]
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "missing_input_file", "observed": ", ".join(missing),
                "threshold": "all four trip files expected",
                "handling_or_note": f"{prefix}: file availability requires review.",
            })
        if row.invalid_coordinate_rows:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "invalid_coordinates_removed",
                "observed": int(row.invalid_coordinate_rows),
                "threshold": "non-finite or outside lon/lat bounds",
                "handling_or_note": "Rows are excluded from trajectory processing.",
            })
        if row.invalid_timestamp_rows:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "invalid_timestamps_removed",
                "observed": int(row.invalid_timestamp_rows),
                "threshold": "non-finite timestamp",
                "handling_or_note": "Rows are excluded; no timestamp is imputed.",
            })
        if row.removed_exact_duplicates:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "consecutive_exact_duplicates_removed",
                "observed": int(row.removed_exact_duplicates),
                "threshold": "identical consecutive lon/lat/timestamp row",
                "handling_or_note": "Only exact adjacent duplicate records are removed.",
            })
        if row.large_gaps_gt_5s > 300:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "many_long_sampling_gaps",
                "observed": int(row.large_gaps_gt_5s),
                "threshold": ">300 intervals longer than 5 seconds",
                "handling_or_note": "Retain the fixes; do not interpolate; invalidate motion context after each gap.",
            })
        if row.max_gap_s > 3600:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "extreme_sampling_gap",
                "observed": float(row.max_gap_s),
                "threshold": ">3600 seconds (1 hour)",
                "handling_or_note": "Likely trip interruption; no path is inferred across this interval.",
            })
        if row.weak_component_count > 100:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "high_weak_component_count",
                "observed": int(row.weak_component_count),
                "threshold": ">100 weak components",
                "handling_or_note": "Inspect graph fragments before path decoding.",
            })
        if row.candidate_miss_rate > 0.10:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "high_candidate_miss_rate",
                "observed": float(row.candidate_miss_rate),
                "threshold": ">10% of observations lack the reference arc among candidates",
                "handling_or_note": "Review candidate generation for this trip; this is coverage, not model accuracy.",
            })
        if row.route_invalid_arc_ids:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "invalid_reference_route_arc_ids",
                "observed": int(row.route_invalid_arc_ids),
                "threshold": "route arc IDs must index .arcs",
                "handling_or_note": "Invalid IDs are excluded from route diagnostics and labels.",
            })
        if row.route_disconnected_transitions:
            anomalies.append({
                "split": row.split, "trip_id": row.trip_id,
                "anomaly": "disconnected_reference_route_arcs",
                "observed": int(row.route_disconnected_transitions),
                "threshold": "consecutive arcs should have matching end/start node IDs",
                "handling_or_note": "Reference sequence is retained but flagged; verify encoding/route limitations.",
            })
    anomaly_columns = [
        "split", "trip_id", "anomaly", "observed", "threshold",
        "handling_or_note",
    ]
    anomaly_frame = pd.DataFrame(anomalies, columns=anomaly_columns)
    anomaly_frame.to_csv(ARTIFACTS / "data_quality_report.csv", index=False)
    anomaly_table = (
        markdown_table(anomaly_frame.head(30))
        if not anomaly_frame.empty
        else "No thresholded data-quality anomalies were detected."
    )

    gap_count_outliers = merged.loc[
        merged["large_gaps_gt_5s"] > 300,
        ["split", "trip_id", "large_gaps_gt_5s", "max_gap_s"],
    ]
    extreme_gap_trips = merged.loc[
        merged["max_gap_s"] > 3600,
        ["split", "trip_id", "max_gap_s", "large_gaps_gt_5s"],
    ]
    max_gap_trip = merged.loc[
        merged["max_gap_s"].idxmax(),
        ["split", "trip_id", "max_gap_s", "large_gaps_gt_5s"],
    ]
    top_miss_trips = merged.sort_values(
        "candidate_miss_rate", ascending=False
    ).head(5)[["split", "trip_id", "candidate_miss_rate"]]
    gap_outlier_table = markdown_table(gap_count_outliers)
    extreme_gap_table = markdown_table(extreme_gap_trips)
    top_miss_table = markdown_table(
        top_miss_trips.assign(
            candidate_miss_rate=top_miss_trips["candidate_miss_rate"].map(
                lambda value: f"{value:.2%}"
            )
        )
    )
    route_disconnect_trips = merged.loc[
        merged["route_disconnected_transitions"] > 0,
        ["split", "trip_id", "route_disconnected_transitions"],
    ]
    route_disconnect_table = (
        markdown_table(route_disconnect_trips)
        if not route_disconnect_trips.empty
        else "None"
    )
    route_transition_total = int(inventory_frame["route_valid_transition_pairs"].sum())
    route_disconnected_total = int(
        inventory_frame["route_disconnected_transitions"].sum()
    )

    report = f"""# Dataset exploration findings

## Dataset scope and partitions

The supplied directory contains the following trip partitions:

{partition_table}

The evaluator-testing split is not present ({int(evaluator['actual_trips'])} trips found; {int(evaluator['expected_trips'])} expected). It is not synthesized or used by this pipeline.

{split_table}

## File inventory and input schema

Every trip is inventoried in `dataset_inventory.csv`, including presence and byte size for `.track`, `.nodes`, `.arcs`, and `.route`, plus total bytes per trip. The current inventory contains {missing_file_count} missing files across those four file types and totals {total_bytes:,} bytes.

All listed files are headerless whitespace-separated text:

{schema_table}

The `.arcs` row index is treated as `arc_id`; `.route` contains the chronological sequence of those arc IDs. This is the supplied encoding, not an independently verified physical ground truth.

## Data quality and preprocessing

The pipeline removes rows with non-finite/out-of-range GPS coordinates or non-finite timestamps, removes consecutive exact duplicate rows, and preserves input order. It fails on decreasing timestamps rather than sorting and potentially changing trajectory order. It does not impute or interpolate GPS positions: no location is inferable across a dropout without additional assumptions.

For these supplied trips, {int(inventory_frame['invalid_coordinate_rows'].sum())} invalid-coordinate rows, {int(inventory_frame['invalid_timestamp_rows'].sum())} invalid-timestamp rows, and {int(inventory_frame['removed_exact_duplicates'].sum())} consecutive exact duplicates were found. {one_second_fraction:.1%} of cleaned time intervals are approximately one second; the median per-trip interval is {median_trip_gap:.2f} seconds. There are {int(inventory_frame['large_gaps_gt_5s'].sum())} intervals over five seconds; the largest is {max_gap:,.0f} seconds. Long gaps remain in the data, are recorded in the quality report, and invalidate motion-derived context immediately after the gap.

The combined coordinate extent is longitude {lon_min:.4f} to {lon_max:.4f} and latitude {lat_min:.4f} to {lat_max:.4f}. This is the union of separate trips across multiple regions, not one continuous operating area. The per-trip extent summary is:

{extent_table}

`gps_coordinate_distributions.png` shows the cleaned longitude and latitude distributions separately for train and student-testing; the multiple separated clusters reflect trips in different regions, not one continuous city-scale extent. `gps_trajectory_patterns.png` shows six selected trajectories (three per partition, spanning trip sizes); line segments are broken at gaps over five seconds rather than drawing an invented path through interruptions.

The maximum recorded gap is in {max_gap_trip['split']}/{max_gap_trip['trip_id']}: {float(max_gap_trip['max_gap_s']):,.0f} seconds ({int(max_gap_trip['large_gaps_gt_5s'])} gaps >5 seconds in that trip).

Distance calculations use an approximate trip-local metre projection, appropriate for local candidate geometry but not intended as a global map projection.

## Road-network structure

The directed graph uses `.arcs` as directed edges from `from_node_id` to `to_node_id`; `.nodes` supplies the coordinates. `network_connectivity.csv` reports per-trip invalid endpoints, isolated nodes, zero in/out-degree nodes, weakly connected components, largest-component share, and strongly connected component count. Weak components treat direction as ignored; strong components require directed reachability.

Across the supplied trip networks, there are {int(inventory_frame['isolated_nodes'].sum()):,} isolated-node instances; the median per-trip isolated-node count is {median_isolates:.0f} (maximum {max_isolates:,}). {network_trips_with_multiple_components} networks have multiple weak components (median {median_weak_components:.0f}, maximum {max_weak_components:,}); {invalid_arc_endpoint_total} invalid arc endpoints occur across {network_trips_with_invalid_arcs} networks. Consult the per-trip CSV for each graph rather than treating networks as one graph.

The static figure `road_network_visualization.png` shows the component-count distribution for both partitions and a component-coloured graph for one training trip.

## Sampling gaps and trip-specific anomalies

`sampling_gap_distribution.png` counts consecutive GPS observation intervals once each (not once per generated candidate), uses a log time axis to retain very long interruptions, and labels train and student-testing separately. The per-trip anomaly list is in `data_quality_report.csv`. Thresholds are explicitly recorded in that file; for example, more than 300 gaps over five seconds and any gap over one hour are flagged.

Trips with more than 300 gaps over five seconds:

{gap_outlier_table}

Trips with a gap over one hour:

{extreme_gap_table}

Trips with the five highest candidate miss rates:

{top_miss_table}

## Candidate features and rationale

Up to 16 nearby road-segment candidates are generated per GPS fix using a spatial index over segment midpoints. Candidate geometries are then projected to compute distance and along-segment position. The model-ready matrix contains:

{feature_table}

`motion_valid` is false for the first observation and after gaps longer than five seconds; speed and bearing are set to neutral values in those rows. The supplied reference route is used only to construct development labels and diagnostics. Route IDs, reference distances, and labels are not included in the exported model input matrix.

The candidate-generation miss rate (reference arc absent from the candidate set) is:

{candidate_coverage_table}

For training, the observation-weighted miss rate is compared against the median per-trip rate in that table (currently about 9.08% weighted versus 3.83% median). A high miss rate means the candidate generator failed to include the heuristic reference arc; it is not a classifier score. Trips over the 10% review threshold are recorded in `data_quality_report.csv`.

## Supplied reference-route limitations

The route file is a sequence of arc IDs, not an observation-aligned label for every GPS fix. Its length therefore need not equal the number of observations. The per-trip route length/observation ratio and directed adjacency checks are in `route_quality.csv`. Across the data, {route_disconnected_total} of {route_transition_total} checkable consecutive route transitions do not join end-to-start; affected trips are listed below. This check is a structural diagnostic, not proof that the supplied route is physically wrong. The pipeline associates observations to reference arcs by nearest geometry, which is a heuristic and can be ambiguous near parallel roads or during GPS gaps.

{route_table}

{route_disconnect_table}

## Exploratory map

The interactive map `{map_path.name if map_path else 'was not generated (Folium unavailable)'}` overlays the sampled GPS trajectory, reference-route arcs, and nearby network context (the 20 nearest segment midpoints for sampled fixes), with a legend and layer control. It is a local visual aid, not a complete rendering of every edge in a large trip network. The static screenshot for report inclusion is `dataset_exploration_map_screenshot.png`.

## Issue and assumption log

The full thresholded anomaly list is in `data_quality_report.csv`; per-trip file sizes, schema, extents, quality, and graph metrics remain in `dataset_inventory.csv`. Main assumptions: coordinates are longitude/latitude; arc row indices define arc IDs; graph edges are interpreted as directed; and gaps over five seconds are too long for local speed/bearing context. No interpolation or map reconstruction is claimed for a gap.

First 30 anomaly records (the CSV contains the full list):

{anomaly_table}
"""
    (ARTIFACTS / "dataset_exploration.md").write_text(report, encoding="utf-8")


def main():
    inventory_frame = inventory()
    feature_frame, summary = process_labeled_splits()
    make_visualizations(inventory_frame, feature_frame)
    folium_map = make_folium_map()
    make_folium_map("00000001", "00000001_route_map.html")
    write_dataset_findings(inventory_frame, summary, folium_map)
    metrics = {
        "train_observations": int(inventory_frame.loc[inventory_frame["split"] == "train", "observations"].sum()),
        "student_test_observations": int(inventory_frame.loc[inventory_frame["split"] == "student_test", "observations"].sum()),
        "candidate_rows": int(len(feature_frame)),
        "positive_candidate_rows": int(feature_frame["label"].sum()),
        "median_candidate_miss_rate": float(summary["candidate_miss_rate"].median()),
        "features": list(FEATURE_COLUMNS),
        "long_gaps_gt_5s_train": int(
            inventory_frame.loc[inventory_frame["split"] == "train", "large_gaps_gt_5s"].sum()
        ),
        "long_gaps_gt_5s_student_test": int(
            inventory_frame.loc[inventory_frame["split"] == "student_test", "large_gaps_gt_5s"].sum()
        ),
        "invalid_coordinate_rows_removed": int(inventory_frame["invalid_coordinate_rows"].sum()),
        "invalid_timestamp_rows_removed": int(inventory_frame["invalid_timestamp_rows"].sum()),
        "exact_duplicates_removed": int(inventory_frame["removed_exact_duplicates"].sum()),
        "folium_map": (
            folium_map.relative_to(ROOT).as_posix() if folium_map else None
        ),
    }
    (ARTIFACTS / "pipeline_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

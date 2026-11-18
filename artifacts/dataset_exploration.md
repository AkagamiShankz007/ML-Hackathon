# Dataset exploration findings

## Dataset scope and partitions

The supplied directory contains the following trip partitions:

| split | expected_trips | actual_trips | available |
| --- | --- | --- | --- |
| train | 80 | 80 | True |
| student_test | 10 | 10 | True |
| evaluator_test | 10 | 0 | False |

The evaluator-testing split is not present (0 trips found; 10 expected). It is not synthesized or used by this pipeline.

| split | trips | raw_observations | cleaned_observations | gps_gaps_gt_5s | median_trip_observations | median_weak_components | trips_with_multiple_weak_components | median_isolated_nodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student_test | 10 | 19160 | 19160 | 44 | 1420.0 | 11.0 | 10 | 0.0 |
| train | 80 | 208417 | 208417 | 1216 | 1529.0 | 20.0 | 79 | 0.0 |

## File inventory and input schema

Every trip is inventoried in `dataset_inventory.csv`, including presence and byte size for `.track`, `.nodes`, `.arcs`, and `.route`, plus total bytes per trip. The current inventory contains 0 missing files across those four file types and totals 94,106,043 bytes.

All listed files are headerless whitespace-separated text:

| File | Column | Meaning | Unit/encoding | Parsed dtype |
| --- | --- | --- | --- | --- |
| .track | 0 | longitude | degrees | float64 |
| .track | 1 | latitude | degrees | float64 |
| .track | 2 | timestamp | seconds | float64 |
| .nodes | 0 | longitude | degrees | float64 |
| .nodes | 1 | latitude | degrees | float64 |
| .arcs | 0 | from_node_id | 0-based node index | int64 |
| .arcs | 1 | to_node_id | 0-based node index | int64 |
| .route | 0 | arc_id | 0-based .arcs row index | int64 |

The `.arcs` row index is treated as `arc_id`; `.route` contains the chronological sequence of those arc IDs. This is the supplied encoding, not an independently verified physical ground truth.

## Data quality and preprocessing

The pipeline removes rows with non-finite/out-of-range GPS coordinates or non-finite timestamps, removes consecutive exact duplicate rows, and preserves input order. It fails on decreasing timestamps rather than sorting and potentially changing trajectory order. It does not impute or interpolate GPS positions: no location is inferable across a dropout without additional assumptions.

For these supplied trips, 0 invalid-coordinate rows, 0 invalid-timestamp rows, and 0 consecutive exact duplicates were found. 96.7% of cleaned time intervals are approximately one second; the median per-trip interval is 1.00 seconds. There are 1260 intervals over five seconds; the largest is 153,064 seconds. Long gaps remain in the data, are recorded in the quality report, and invalidate motion-derived context immediately after the gap.

The combined coordinate extent is longitude -100.4185 to 137.2370 and latitude -34.6403 to 61.6996. This is the union of separate trips across multiple regions, not one continuous operating area. The per-trip extent summary is:

| split | trips | median_longitude_span_km | p90_longitude_span_km | median_latitude_span_km | p90_latitude_span_km |
| --- | --- | --- | --- | --- | --- |
| student_test | 10 | 5.22 | 17.14 | 5.91 | 20.1 |
| train | 80 | 7.7 | 38.17 | 5.25 | 29.83 |

`gps_coordinate_distributions.png` shows the cleaned longitude and latitude distributions separately for train and student-testing; the multiple separated clusters reflect trips in different regions, not one continuous city-scale extent. `gps_trajectory_patterns.png` shows six selected trajectories (three per partition, spanning trip sizes); line segments are broken at gaps over five seconds rather than drawing an invented path through interruptions.

The maximum recorded gap is in train/00000090: 153,064 seconds (20 gaps >5 seconds in that trip).

Distance calculations use an approximate trip-local metre projection, appropriate for local candidate geometry but not intended as a global map projection.

## Road-network structure

The directed graph uses `.arcs` as directed edges from `from_node_id` to `to_node_id`; `.nodes` supplies the coordinates. `network_connectivity.csv` reports per-trip invalid endpoints, isolated nodes, zero in/out-degree nodes, weakly connected components, largest-component share, and strongly connected component count. Weak components treat direction as ignored; strong components require directed reachability.

Across the supplied trip networks, there are 0 isolated-node instances; the median per-trip isolated-node count is 0 (maximum 0). 89 networks have multiple weak components (median 19, maximum 421); 0 invalid arc endpoints occur across 0 networks. Consult the per-trip CSV for each graph rather than treating networks as one graph.

The static figure `road_network_visualization.png` shows the component-count distribution for both partitions and a component-coloured graph for one training trip.

## Sampling gaps and trip-specific anomalies

`sampling_gap_distribution.png` counts consecutive GPS observation intervals once each (not once per generated candidate), uses a log time axis to retain very long interruptions, and labels train and student-testing separately. The per-trip anomaly list is in `data_quality_report.csv`. Thresholds are explicitly recorded in that file; for example, more than 300 gaps over five seconds and any gap over one hour are flagged.

Trips with more than 300 gaps over five seconds:

| split | trip_id | large_gaps_gt_5s | max_gap_s |
| --- | --- | --- | --- |
| train | 00000004 | 320 | 51.0 |

Trips with a gap over one hour:

| split | trip_id | max_gap_s | large_gaps_gt_5s |
| --- | --- | --- | --- |
| train | 00000078 | 52810.0 | 124 |
| train | 00000090 | 153064.0 | 20 |

Trips with the five highest candidate miss rates:

| split | trip_id | candidate_miss_rate |
| --- | --- | --- |
| train | 00000033 | 66.14% |
| train | 00000054 | 37.10% |
| train | 00000003 | 17.17% |
| train | 00000079 | 17.13% |
| train | 00000001 | 15.74% |

## Candidate features and rationale

Up to 16 nearby road-segment candidates are generated per GPS fix using a spatial index over segment midpoints. Candidate geometries are then projected to compute distance and along-segment position. The model-ready matrix contains:

| Feature | Reason for inclusion |
| --- | --- |
| distance_m | Measures spatial fit between an observation and a candidate road segment. |
| log_distance | Compresses the long-tailed distance range while retaining proximity ordering. |
| projection | Locates the GPS point along the candidate segment, from its start to its end. |
| segment_length_m | Adds candidate geometry context; longer segments can behave differently. |
| log_segment_length | Compresses the skewed segment-length range. |
| bearing_delta | Measures directional agreement between recent GPS movement and the road segment. |
| gps_speed_mps | Provides short-interval movement context when the previous fix is reliable. |
| log_gps_gap_s | Represents irregular sampling gaps with a compressed scale. |
| motion_valid | Flags whether speed/bearing context is usable (not the first fix or after a >5 s gap). |

`motion_valid` is false for the first observation and after gaps longer than five seconds; speed and bearing are set to neutral values in those rows. The supplied reference route is used only to construct development labels and diagnostics. Route IDs, reference distances, and labels are not included in the exported model input matrix.

The candidate-generation miss rate (reference arc absent from the candidate set) is:

| split | median_trip_candidate_miss_rate | observation_weighted_candidate_miss_rate |
| --- | --- | --- |
| student_test | 4.40% | 4.61% |
| train | 3.83% | 9.08% |

For training, the observation-weighted miss rate is compared against the median per-trip rate in that table (currently about 9.08% weighted versus 3.83% median). A high miss rate means the candidate generator failed to include the heuristic reference arc; it is not a classifier score. Trips over the 10% review threshold are recorded in `data_quality_report.csv`.

## Supplied reference-route limitations

The route file is a sequence of arc IDs, not an observation-aligned label for every GPS fix. Its length therefore need not equal the number of observations. The per-trip route length/observation ratio and directed adjacency checks are in `route_quality.csv`. Across the data, 14 of 34723 checkable consecutive route transitions do not join end-to-start; affected trips are listed below. This check is a structural diagnostic, not proof that the supplied route is physically wrong. The pipeline associates observations to reference arcs by nearest geometry, which is a heuristic and can be ambiguous near parallel roads or during GPS gaps.

| Partition | Median route/observation ratio | Minimum ratio | Maximum ratio | Connected transitions | Disconnected transitions | Trips with discontinuities | Repeated arc visits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student_test | 0.22 | 0.069 | 0.409 | 3466 | 1 | 1 | 255 |
| train | 0.189 | 0.018 | 0.793 | 31243 | 13 | 10 | 67 |

| split | trip_id | route_disconnected_transitions |
| --- | --- | --- |
| train | 00000002 | 1 |
| train | 00000019 | 3 |
| train | 00000033 | 1 |
| train | 00000041 | 1 |
| train | 00000042 | 1 |
| train | 00000069 | 1 |
| train | 00000075 | 2 |
| train | 00000080 | 1 |
| train | 00000086 | 1 |
| train | 00000089 | 1 |
| student_test | 00000076 | 1 |

## Exploratory map

The interactive map `dataset_exploration_map.html` overlays the sampled GPS trajectory, reference-route arcs, and nearby network context (the 20 nearest segment midpoints for sampled fixes), with a legend and layer control. It is a local visual aid, not a complete rendering of every edge in a large trip network. The static screenshot for report inclusion is `dataset_exploration_map_screenshot.png`.

## Issue and assumption log

The full thresholded anomaly list is in `data_quality_report.csv`; per-trip file sizes, schema, extents, quality, and graph metrics remain in `dataset_inventory.csv`. Main assumptions: coordinates are longitude/latitude; arc row indices define arc IDs; graph edges are interpreted as directed; and gaps over five seconds are too long for local speed/bearing context. No interpolation or map reconstruction is claimed for a gap.

First 30 anomaly records (the CSV contains the full list):

| split | trip_id | anomaly | observed | threshold | handling_or_note |
| --- | --- | --- | --- | --- | --- |
| train | 00000001 | high_weak_component_count | 175.0 | >100 weak components | Inspect graph fragments before path decoding. |
| train | 00000001 | high_candidate_miss_rate | 0.1574144486692015 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000002 | disconnected_reference_route_arcs | 1.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000003 | high_candidate_miss_rate | 0.17174177831912302 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000004 | many_long_sampling_gaps | 320.0 | >300 intervals longer than 5 seconds | Retain the fixes; do not interpolate; invalidate motion context after each gap. |
| train | 00000018 | high_candidate_miss_rate | 0.10554089709762533 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000019 | disconnected_reference_route_arcs | 3.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000028 | high_candidate_miss_rate | 0.11809256661991585 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000029 | high_candidate_miss_rate | 0.10914051841746249 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000031 | high_candidate_miss_rate | 0.1314487632508834 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000032 | high_weak_component_count | 126.0 | >100 weak components | Inspect graph fragments before path decoding. |
| train | 00000033 | high_candidate_miss_rate | 0.6613620807665982 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000033 | disconnected_reference_route_arcs | 1.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000035 | high_weak_component_count | 119.0 | >100 weak components | Inspect graph fragments before path decoding. |
| train | 00000041 | disconnected_reference_route_arcs | 1.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000042 | disconnected_reference_route_arcs | 1.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000043 | high_weak_component_count | 421.0 | >100 weak components | Inspect graph fragments before path decoding. |
| train | 00000043 | high_candidate_miss_rate | 0.15460809646856158 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000054 | high_weak_component_count | 213.0 | >100 weak components | Inspect graph fragments before path decoding. |
| train | 00000054 | high_candidate_miss_rate | 0.37099363123284834 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000055 | high_candidate_miss_rate | 0.15210355987055016 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000069 | disconnected_reference_route_arcs | 1.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000074 | high_weak_component_count | 118.0 | >100 weak components | Inspect graph fragments before path decoding. |
| train | 00000075 | disconnected_reference_route_arcs | 2.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000078 | extreme_sampling_gap | 52810.0 | >3600 seconds (1 hour) | Likely trip interruption; no path is inferred across this interval. |
| train | 00000079 | high_candidate_miss_rate | 0.17126269956458637 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |
| train | 00000080 | disconnected_reference_route_arcs | 1.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000086 | high_weak_component_count | 193.0 | >100 weak components | Inspect graph fragments before path decoding. |
| train | 00000086 | disconnected_reference_route_arcs | 1.0 | consecutive arcs should have matching end/start node IDs | Reference sequence is retained but flagged; verify encoding/route limitations. |
| train | 00000089 | high_candidate_miss_rate | 0.11074918566775244 | >10% of observations lack the reference arc among candidates | Review candidate generation for this trip; this is coverage, not model accuracy. |

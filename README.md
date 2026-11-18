# PathSnap

Independent data-processing and feature-engineering pipeline for the PathSnap
map-matching dataset.

## Pipeline

Run:

```bash
python -m pip install -r requirements.txt
python src/build_features.py
```

The pipeline:

1. Inventories every trip in `train/` and `student_test/`.
2. Validates GPS coordinates and timestamp ordering.
3. Converts longitude/latitude differences to approximate local metres.
4. Builds spatial candidates using arc geometry and the supplied reference arc
   sequence.
5. Computes point-to-segment, motion, bearing, gap, and route-membership features.
6. Writes CSV, NPZ, JSON, plot, and Folium deliverables under `artifacts/`.

Evaluator-testing trips are inventory-only. Labels and routes from that
partition are never read or used.

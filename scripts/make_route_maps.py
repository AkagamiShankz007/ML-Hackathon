"""Interactive Folium maps (GPS fixes, road network, reference route, classifier-only route, classifier + HMM route) for student-test trips,
plus PNG screenshots of them (Playwright + installed Chrome) for the report.
Usage: python scripts/make_route_maps.py [trip_id ...]      default: 00000077 (HMM helps) 00000052 (HMM does not help Top-1) 00000007
Writes artifacts/hmm/route_reconstruction_trip_<id>.html and artifacts/hmm/route_reconstruction_trip_<id>.png"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import hmm_tuning as HV, hmm_decoder as H

trips_wanted = sys.argv[1:] or ["00000077", "00000052", "00000007"]
params = json.loads((ROOT / "artifacts" / "hmm" / "hmm_config.json").read_text())["decoder_params"]
trips = HV.prepare(HV.attach_geometry(pd.read_parquet(ROOT / "artifacts" / "cache" / "test_logreg.parquet"), False), "student_test")
out = ROOT / "artifacts" / "hmm"
for tid in trips_wanted:
    t = trips[tid]; k_clf = t["Z"].argmax(1); k_hmm = H.decode(t["Z"], t["cls"], t["short"], params, t["aux"])
    HV.make_map(out / f"route_reconstruction_trip_{tid}.html", tid, t, k_clf, k_hmm); print("map:", tid)
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True); pg = b.new_page(viewport={"width": 1400, "height": 850})
    for tid in trips_wanted:
        pg.goto((out / f"route_reconstruction_trip_{tid}.html").resolve().as_uri()); pg.wait_for_timeout(6000)
        pg.screenshot(path=out / f"route_reconstruction_trip_{tid}.png"); print("screenshot:", tid)
    b.close()

"""End-to-end API check: every served trip, both modes, through HTTP (what the evaluator does on the demo day).
Usage: python scripts/check_api_all_trips.py [--api http://localhost:8000]
Writes artifacts/api_validation/all_trips_results.csv and checks that no reference data is exposed in the responses."""
import argparse, csv, json, time
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("reference_route", "reference_arc", "ground_truth", "label")     # keys that must never appear in a response


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def keys(o, acc=None):
    acc = set() if acc is None else acc
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(k); keys(v, acc)
    elif isinstance(o, list) and o and isinstance(o[0], (dict, list)):
        keys(o[0], acc)
    return acc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--api", default="http://localhost:8000"); a = ap.parse_args()
    health = call(a.api + "/health"); print("health:", health)
    trips = call(a.api + "/trips")["trips"]; rows = []
    for t in trips:
        t0 = time.time(); r = call(a.api + "/reconstruct", dict(trip_id=t["trip_id"], mode="hmm", compare=True)); dt = time.time() - t0
        leaked = [k for k in keys(r) if k in FORBIDDEN]
        c = r["comparison"] or {}; h, k = c.get("hmm") or {}, c.get("classifier") or {}
        rows.append(dict(trip_id=t["trip_id"], observations=r["n_observations_used"], matched=r["n_matched"], segment_breaks=r["n_segment_breaks"],
                         route_arcs=len(r["route_arc_ids"]), top1_classifier=k.get("top1_accuracy"), top1_hmm=h.get("top1_accuracy"),
                         continuity_classifier=k.get("route_continuity"), continuity_hmm=h.get("route_continuity"), seq_f1_hmm=h.get("seq_f1"),
                         edit_similarity_hmm=h.get("edit_similarity"), server_ms=round(r["timings_ms"]["request_total_ms"]), client_s=round(dt, 2), leaked_keys=";".join(leaked)))
        print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rows[-1].items()}, flush=True)
    out = ROOT / "artifacts" / "api_validation"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "all_trips_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = sum(r["observations"] for r in rows)
    print(f"{len(rows)} trips OK | observation-weighted Top-1: classifier {sum(r['top1_classifier']*r['observations'] for r in rows)/n:.4f} -> HMM {sum(r['top1_hmm']*r['observations'] for r in rows)/n:.4f}"
          f" | leaked keys: {sorted({k for r in rows for k in r['leaked_keys'].split(';') if k}) or 'none'}")


if __name__ == "__main__":
    main()

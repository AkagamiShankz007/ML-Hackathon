"""PathSnap evaluation API (FastAPI).

  uvicorn api.main:app --host 0.0.0.0 --port 8000        (from the repository root)

Environment
  PATHSNAP_DATA_DIR            folder that contains the partition folders          (default: "Student folder")
  PATHSNAP_SERVE_PARTITION     partition served to the UI, e.g. student_test / evaluator_test   (default: student_test)
  PATHSNAP_MODEL_DIR           exported bundle                                     (default: artifacts/final_model)
  PATHSNAP_CORS_ORIGINS        comma-separated allowed UI origins                  (default: *)

Each partition folder holds one sub-folder per trip: <trip>/<trip>.track, .nodes, .arcs and (optionally) .route. The route
file is only read inside the server to compute the metrics; no response field contains reference-route data."""
from __future__ import annotations
import logging, os, re, sys, time, uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pathsnap_inference as P                                   # noqa: E402

DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT / "Student folder"))
PARTITION = os.environ.get("PATHSNAP_SERVE_PARTITION", "student_test")
MODEL_DIR = Path(os.environ.get("PATHSNAP_MODEL_DIR", ROOT / "artifacts" / "final_model"))
CORS = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",")]

LOG_DIR = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api"); log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")):
        h.setFormatter(fmt); log.addHandler(h)

app = FastAPI(title="PathSnap API", version="1.0",
              description="GPS map matching: Bayesian-tuned logistic regression + HMM/Viterbi route reconstruction")
app.add_middleware(CORSMiddleware, allow_origins=CORS, allow_methods=["*"], allow_headers=["*"])

STATE: dict = {"bundle": None, "trips": {}, "messages": []}
TRIP_ID = re.compile(r"^[0-9A-Za-z_-]+$")


class ReconstructRequest(BaseModel):
    trip_id: str
    mode: Literal["hmm", "classifier"] = "hmm"
    compare: bool = False


@app.on_event("startup")
def startup():
    msgs = []
    try:
        STATE["bundle"] = P.load_bundle(MODEL_DIR)
        log.info("model loaded: %s from %s", STATE["bundle"].model_name, MODEL_DIR)
    except Exception as e:                                       # keep serving /health with an explanation
        msgs.append(f"Model could not be loaded from {MODEL_DIR}: {e}")
    part = DATA_DIR / PARTITION
    if not part.is_dir():
        msgs.append(f"Partition folder not found: {part}")
    else:
        for d in sorted(part.iterdir()):
            if d.is_dir() and TRIP_ID.match(d.name) and all((d / f"{d.name}.{x}").exists() for x in ("track", "nodes", "arcs")):
                try:
                    STATE["trips"][d.name] = P.trip_summary(d, d.name)
                except Exception as e:
                    msgs.append(f"Trip {d.name} skipped: {e}")
    STATE["messages"] = msgs
    log.info("serving partition=%s trips=%d", PARTITION, len(STATE["trips"]))


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid, t0 = uuid.uuid4().hex[:8], time.perf_counter()
    log.info("[%s] %s %s from %s", rid, request.method, request.url.path, request.client.host if request.client else "?")
    try:
        resp = await call_next(request)
    except Exception:
        log.exception("[%s] unhandled error", rid)
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Internal server error"}, status_code=500)
    log.info("[%s] %s %s -> %d (%.0f ms)", rid, request.method, request.url.path, resp.status_code, (time.perf_counter() - t0) * 1000)
    return resp


def trip_dir(trip_id: str) -> Path:
    if not TRIP_ID.match(trip_id) or trip_id not in STATE["trips"]:
        raise HTTPException(404, f"Unknown trip_id '{trip_id}'")
    return DATA_DIR / PARTITION / trip_id


@app.get("/health")
def health():
    ok = STATE["bundle"] is not None and bool(STATE["trips"])
    b = STATE["bundle"]
    return dict(status="ok" if ok else "degraded", model_loaded=b is not None, model_name=b.model_name if b else None,
                emission=b.emission if b else None, served_partition=PARTITION, n_trips=len(STATE["trips"]), messages=STATE["messages"])


@app.get("/trips")
def trips():
    return dict(partition=PARTITION, trips=list(STATE["trips"].values()))


@lru_cache(maxsize=32)
def _network(trip_id: str):
    return P.network_near_trip(trip_dir(trip_id), trip_id)


@app.get("/trips/{trip_id}/network")
def network(trip_id: str):
    trip_dir(trip_id)
    try:
        arcs = _network(trip_id)
    except Exception as e:
        raise HTTPException(422, f"Trip data could not be processed: {e}")
    return dict(trip_id=trip_id, n_arcs=len(arcs), arcs=arcs)


@app.post("/reconstruct")
def reconstruct(req: ReconstructRequest, request: Request):
    d = trip_dir(req.trip_id)
    if STATE["bundle"] is None:
        raise HTTPException(503, "Model not loaded")
    log.info("POST /reconstruct trip=%s mode=%s compare=%s", req.trip_id, req.mode, req.compare)
    try:
        r = P.reconstruct(STATE["bundle"], d, req.trip_id, req.mode, req.compare)
    except (ValueError, KeyError, IndexError, OSError) as e:
        raise HTTPException(422, f"Trip data could not be processed: {e}")
    m = r["metrics"] or {}
    log.info("reconstruct trip=%s mode=%s model=%s obs=%d matched=%d breaks=%d top1=%s continuity=%s total=%.0fms", req.trip_id, req.mode,
             r["model_name"], r["n_observations_used"], r["n_matched"], r["n_segment_breaks"], m.get("top1_accuracy"), m.get("route_continuity"),
             r["timings_ms"]["request_total_ms"])
    return r

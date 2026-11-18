#!/usr/bin/env bash
# LAPTOP 1 (model + API).  Usage: ./scripts/run_api.sh [student_test|evaluator_test] [data_dir]
cd "$(dirname "$0")/.."
export PATHSNAP_SERVE_PARTITION="${1:-student_test}"
[ -n "$2" ] && export PATHSNAP_DATA_DIR="$2"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

#!/usr/bin/env bash
# Usage: ./run.sh [student_test|evaluator_test]   -> API on :8000, UI on :5173
export PATHSNAP_SERVE_PARTITION="${1:-student_test}"
(cd "$(dirname "$0")" && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000) &
(cd "$(dirname "$0")/frontend" && python -m http.server 5173) &
wait

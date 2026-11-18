#!/usr/bin/env bash
# LAPTOP 2 (user interface).  Usage: ./scripts/run_ui.sh <ip-of-laptop-1> [port]
cd "$(dirname "$0")/../frontend"
echo "Open http://localhost:5173/?api=http://$1:${2:-8000}"
python -m http.server 5173

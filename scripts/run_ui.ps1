# LAPTOP 2 (user interface).  Usage: .\scripts\run_ui.ps1 <ip-of-laptop-1> [port]
# Serves the static RouteLens UI on :5173 and opens it pointed at the API of laptop 1.
param([Parameter(Mandatory = $true)][string]$ApiHost, [int]$ApiPort = 8000)
$root = Split-Path -Parent $PSScriptRoot
Start-Process "http://localhost:5173/?api=http://${ApiHost}:${ApiPort}"
Set-Location "$root\frontend"
python -m http.server 5173

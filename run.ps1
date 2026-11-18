# Usage: .\run.ps1 [student_test|evaluator_test]
#   API  -> http://localhost:8000   (Swagger UI at /docs)
#   UI   -> http://localhost:5173   (RouteLens, static files from .\frontend)
param([string]$Partition = "student_test")
$root = $PSScriptRoot
$env:PATHSNAP_SERVE_PARTITION = $Partition
Start-Process -WorkingDirectory $root python -ArgumentList "-m uvicorn api.main:app --host 0.0.0.0 --port 8000"
Start-Process -WorkingDirectory "$root\frontend" python -ArgumentList "-m http.server 5173"
Start-Process "http://localhost:5173"

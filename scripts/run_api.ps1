# LAPTOP 1 (model + API).  Usage: .\scripts\run_api.ps1 [student_test|evaluator_test] [data_dir]
# Serves the trips of <data_dir>\<partition>\<trip>\<trip>.{track,nodes,arcs[,route]} on 0.0.0.0:8000 and logs every request
# to the console and logs\api.log (show this console to the evaluator).  Allow inbound TCP 8000 in the Windows firewall.
param([string]$Partition = "student_test", [string]$DataDir = "")
$root = Split-Path -Parent $PSScriptRoot
$env:PATHSNAP_SERVE_PARTITION = $Partition
if ($DataDir -ne "") { $env:PATHSNAP_DATA_DIR = $DataDir }
Set-Location $root
Write-Host "PathSnap API: partition=$Partition  http://$((Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress):8000/docs"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

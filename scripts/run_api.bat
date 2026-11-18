@echo off
rem LAPTOP 1 - PathSnap API (model + HMM/Viterbi + road network). Usage: scripts\run_api.bat [student_test|evaluator_test]
title PathSnap API (Laptop 1)
cd /d "%~dp0.."
if "%~1"=="" (set PATHSNAP_SERVE_PARTITION=student_test) else (set PATHSNAP_SERVE_PARTITION=%~1)
echo PathSnap API - partition %PATHSNAP_SERVE_PARTITION% - listening on 0.0.0.0:8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause

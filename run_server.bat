@echo off
title PathSnap Backend Server (Laptop 1)
echo ========================================================
echo Starting PathSnap Server (Laptop 1 API)
echo NOTE: KEEP THIS WINDOW OPEN IN THE BACKGROUND!
echo Open your browser to: http://localhost:8000/ui
echo ========================================================
start http://localhost:8000/ui
"C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn src.api:app --host 0.0.0.0 --port 8000
pause

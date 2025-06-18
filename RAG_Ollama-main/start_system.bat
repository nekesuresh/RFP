@echo off
echo Multi-Agent RFP Assistant
========================
echo.
echo Starting the system...
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "python start_backend.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
start "Frontend Server" cmd /k "python start_frontend.py"

echo.
echo System started!
echo.
echo Frontend: http://localhost:8501
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit this window...
pause > nul 
@echo off
REM 🎓 STUDENT ENGAGEMENT MONITORING SYSTEM
REM The BEST and ONLY option you need!

echo ================================================================================
echo 🎓 STUDENT ENGAGEMENT MONITORING SYSTEM - BEST VERSION
echo ================================================================================
echo 📅 Date: %date% %time%
echo 📂 Directory: %cd%
echo 🚀 Launching the BEST monitoring experience...
echo ================================================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Run the BEST monitoring system
echo 🚀 Starting optimized student monitoring...
python student_monitor.py

REM Keep window open
pause
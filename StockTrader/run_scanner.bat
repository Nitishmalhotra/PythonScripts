@echo off
REM Automated Stock Scanner - Windows Batch Script
REM Place this in Windows Task Scheduler for automation

echo ================================================
echo  Stock Scanner - Automated Run
echo ================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Load environment variables using Python
python -c "from dotenv import load_dotenv; load_dotenv()" 2>nul

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the scanner
echo Running scanner...
python Active_Production\automated_scanner.py

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================
    echo  Scanner completed successfully!
    echo ================================================
) else (
    echo.
    echo ================================================
    echo  Scanner failed with error code %ERRORLEVEL%
    echo ================================================
)

REM Optional: Keep window open to see results
REM pause

exit /b %ERRORLEVEL%

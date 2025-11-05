@echo off
REM Local Dependency Verification Script for Windows
echo ============================================================
echo LOCAL DEPENDENCY VERIFICATION
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if requirements.txt exists
if not exist "Configuration\requirements.txt" (
    echo ERROR: Configuration\requirements.txt not found!
    pause
    exit /b 1
)

echo Step 1: Installing/Updating dependencies...
echo --------------------------------------------
pip install -r Configuration\requirements.txt
if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed to install
    echo Continuing with verification...
    echo.
) else (
    echo.
    echo Dependencies installed successfully!
    echo.
)

echo Step 2: Verifying dependencies...
echo --------------------------------------------
python verify_dependencies.py
if errorlevel 1 (
    echo.
    echo Some dependencies are missing. Please check the output above.
    pause
    exit /b 1
)

echo.
echo Step 3: Testing project imports...
echo --------------------------------------------
python test_imports.py
if errorlevel 1 (
    echo.
    echo Some imports failed. Please check the output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo VERIFICATION COMPLETE!
echo ============================================================
echo.
echo Your project dependencies are verified and ready to use.
pause


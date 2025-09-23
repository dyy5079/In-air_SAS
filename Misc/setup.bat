@echo off
REM AirSAS Python Setup Script for Windows
REM Automated installation script for the AirSAS Python environment

setlocal enabledelayedexpansion

echo.
echo 🚀 AirSAS Python Setup Script for Windows
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Found Python %PYTHON_VERSION%

REM Check if pip is available
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found. Please reinstall Python with pip included.
    pause
    exit /b 1
)

echo [INFO] pip is available

REM Create virtual environment
set VENV_NAME=airsas_env
if exist "%VENV_NAME%" (
    echo [WARNING] Virtual environment %VENV_NAME% already exists
    set /p "REPLY=Do you want to remove it and create a new one? [y/N]: "
    if /i "!REPLY!"=="y" (
        echo [INFO] Removing existing virtual environment...
        rmdir /s /q "%VENV_NAME%"
    ) else (
        echo [INFO] Using existing virtual environment
        goto activate_env
    )
)

echo [INFO] Creating virtual environment: %VENV_NAME%
python -m venv %VENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

:activate_env
echo [INFO] Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install core dependencies
echo [INFO] Installing core scientific packages...
pip install numpy>=1.21.0
if errorlevel 1 (
    echo [ERROR] Failed to install numpy
    pause
    exit /b 1
)

pip install scipy>=1.7.0
if errorlevel 1 (
    echo [ERROR] Failed to install scipy
    pause
    exit /b 1
)

pip install matplotlib>=3.5.0
if errorlevel 1 (
    echo [ERROR] Failed to install matplotlib
    pause
    exit /b 1
)

pip install h5py>=3.1.0
if errorlevel 1 (
    echo [ERROR] Failed to install h5py
    pause
    exit /b 1
)

pip install pandas>=1.3.0
if errorlevel 1 (
    echo [ERROR] Failed to install pandas
    pause
    exit /b 1
)

REM Install from requirements.txt if it exists
if exist requirements.txt (
    echo [INFO] Installing from requirements.txt...
    pip install -r requirements.txt
)

echo [SUCCESS] Python dependencies installed

REM Run tests
echo [INFO] Running installation tests...

REM Test basic imports
python -c "import numpy as np; import scipy; import matplotlib.pyplot as plt; import h5py; import pandas as pd; print('✓ Core packages imported successfully'); print('  NumPy version:', np.__version__); print('  SciPy version:', scipy.__version__)"
if errorlevel 1 (
    echo [ERROR] Core package import test failed
    pause
    exit /b 1
)

REM Test AirSAS utilities if available
if exist test_utilities.py (
    echo [INFO] Running AirSAS utilities test...
    python test_utilities.py
    if errorlevel 1 (
        echo [WARNING] Utilities test failed, but core installation is complete
    )
) else (
    echo [WARNING] test_utilities.py not found. Skipping utilities test.
)

echo.
echo 🎉 Installation completed successfully!
echo.
echo To use the AirSAS Python environment:
echo.
echo 1. Activate the virtual environment:
echo    %VENV_NAME%\Scripts\activate.bat
echo.
echo 2. Run the main script:
echo    python makeSasImage.py
echo.
echo 3. To deactivate when done:
echo    deactivate
echo.
echo 📖 See INSTALLATION.md for detailed documentation
echo 🐛 Report issues at: https://github.com/dyy5079/In-air_SAS/issues
echo.

pause
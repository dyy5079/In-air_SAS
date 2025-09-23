# AirSAS Python Installation Guide

This guide provides comprehensive instructions for installing and setting up the AirSAS Python environment.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation Steps](#detailed-installation-steps)
4. [Dependency Management](#dependency-management)
5. [Verification and Testing](#verification-and-testing)
6. [Troubleshooting](#troubleshooting)
7. [Development Setup](#development-setup)

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), Windows 10+
- **Python**: 3.8 or higher (recommended: 3.10+)
- **Memory**: 4GB RAM minimum, 8GB+ recommended for large datasets
- **Storage**: 2GB free space for dependencies and data

### Supported Python Versions
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10 (recommended)
- ✅ Python 3.11
- ✅ Python 3.12

## Quick Installation

### Option 1: Automated Setup Script
```bash
# Clone the repository
git clone https://github.com/dyy5079/In-air_SAS.git
cd In-air_SAS

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual pip Installation
```bash
# Create virtual environment
python3 -m venv airsas_env
source airsas_env/bin/activate  # On Windows: airsas_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_utilities.py
```

## Detailed Installation Steps

### Step 1: Install Python and pip

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python 3.10 and pip
sudo apt install -y python3.10 python3.10-venv python3-pip python3.10-dev

# Install build essentials (for compiling some packages)
sudo apt install -y build-essential
```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.10

# Or using MacPorts
sudo port install python310
```

#### Windows
1. Download Python from [python.org](https://python.org)
2. Run installer with "Add Python to PATH" checked
3. Open Command Prompt or PowerShell as Administrator

### Step 2: Create Virtual Environment
```bash
# Navigate to the project directory
cd /path/to/In-air_SAS

# Create virtual environment
python3 -m venv airsas_env

# Activate virtual environment
# Linux/macOS:
source airsas_env/bin/activate
# Windows:
airsas_env\Scripts\activate

# Verify activation (should show virtual env path)
which python
```

### Step 3: Install Dependencies

#### Core Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install core scientific packages
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0
pip install h5py>=3.1.0
pip install pandas>=1.3.0
```

#### All Dependencies at Once
```bash
# Install from requirements file
pip install -r requirements.txt
```

## Dependency Management

### Core Package Versions
The following versions are tested and confirmed working:

```txt
numpy==2.2.6
scipy==1.15.3
matplotlib==3.10.6
h5py==3.14.0
pandas==2.3.2
```

### Optional Dependencies
```bash
# For Jupyter notebook support
pip install jupyter notebook ipython

# For development and testing
pip install pytest black flake8

# For performance (optional)
pip install numba
```

### Platform-Specific Notes

#### Linux (Ubuntu/Debian)
```bash
# Additional system packages that may be needed
sudo apt install -y libhdf5-dev libffi-dev libssl-dev

# For matplotlib backend support
sudo apt install -y python3-tk
```

#### macOS
```bash
# May need to install command line tools
xcode-select --install

# For HDF5 support with Homebrew
brew install hdf5
```

#### Windows
```bash
# Use pip only - no additional system packages needed
# Consider using Anaconda for easier package management
```

## Verification and Testing

### Basic Installation Test
```bash
# Test Python environment
python --version

# Test core imports
python -c "
import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py
import pandas as pd
print('✓ All core packages imported successfully')
print('NumPy version:', np.__version__)
print('SciPy version:', scipy.__version__)
"
```

### AirSAS Package Test
```bash
# Run the comprehensive test suite
python test_utilities.py

# Expected output:
# === AirSAS Utilities Package Test ===
# Testing utilities package imports...
# ✓ All package imports successful
# Testing basic functionality...
# ✓ sas_colormap() returns array of shape: (256, 3)
# ✓ initStruct(2) returns list of length: 2
# ✓ All tests passed! Package structure is working correctly.
```

### Test Main Script (without data)
```bash
# Test main script imports and basic functionality
python -c "
import sys
sys.path.append('.')
from makeSasImage import *
print('✓ Main script imports successful')
"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. ImportError: No module named 'numpy'
```bash
# Solution: Install numpy
pip install numpy

# If still failing, check Python path
python -c "import sys; print(sys.path)"
```

#### 2. ImportError: cannot import name 'tukey' from 'scipy.signal'
```bash
# Solution: Update scipy version
pip install --upgrade scipy>=1.7.0

# Verify installation
python -c "from scipy.signal.windows import tukey; print('✓ tukey import works')"
```

#### 3. H5py installation fails
```bash
# Ubuntu/Debian
sudo apt install libhdf5-dev

# macOS
brew install hdf5

# Then reinstall h5py
pip install --no-cache-dir h5py
```

#### 4. Matplotlib display issues
```bash
# Linux: Install tkinter backend
sudo apt install python3-tk

# Set backend in Python
python -c "
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
print('✓ Matplotlib backend set')
"
```

#### 5. Permission errors during installation
```bash
# Use --user flag for user-only installation
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### Platform-Specific Issues

#### Windows
- Use `python` instead of `python3`
- Use `Scripts\activate` instead of `bin/activate`
- Consider using Anaconda for easier package management

#### macOS (Apple Silicon)
```bash
# For M1/M2 Macs, may need specific builds
pip install --only-binary=all numpy scipy matplotlib
```

## Development Setup

### For Contributors and Developers

#### 1. Install Development Dependencies
```bash
# Install all dependencies including dev tools
pip install -r requirements-dev.txt

# Or manually:
pip install pytest black flake8 mypy pre-commit
```

#### 2. Set up Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

#### 3. Code Formatting
```bash
# Format code with black
black *.py utilities/*.py

# Check style with flake8
flake8 *.py utilities/*.py
```

#### 4. Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=utilities

# Run specific test
python test_utilities.py
```

### IDE Setup

#### VS Code
Install recommended extensions:
- Python
- Pylance
- Black Formatter
- Python Test Explorer

#### PyCharm
1. Open project directory
2. Configure Python interpreter to use virtual environment
3. Enable code inspection and formatting

## Data Setup

### Sample Data Download
```bash
# Create data directory structure
mkdir -p data/scenes
mkdir -p "data/characterization data"

# Download sample data (replace with actual URLs)
# wget -O data/scenes/t1e4_01.h5 "https://example.com/sample_data.h5"
```

### Configuration
```python
# Update paths in makeSasImage.py
dataFolder = 'data'  # Path to your data directory
filename = 't1e4_01.h5'  # Your HDF5 data file
```

## Environment Variables

### Optional Environment Configuration
```bash
# Add to ~/.bashrc or ~/.zshrc for convenience
export AIRSAS_DATA_PATH="/path/to/your/data"
export PYTHONPATH="${PYTHONPATH}:/path/to/In-air_SAS"
```

## Performance Optimization

### For Large Datasets
```bash
# Install performance libraries
pip install numba  # Just-in-time compilation
pip install dask    # Parallel computing
```

### Memory Management
```python
# In your Python scripts
import gc
gc.collect()  # Force garbage collection when working with large arrays
```

## Getting Help

### Resources
- 📖 [README](README.md) - Basic usage instructions
- 🐛 [Issues](https://github.com/dyy5079/In-air_SAS/issues) - Report bugs
- 📧 Contact: [Your email]

### Community
- Stack Overflow: Tag your questions with `airsas` and `python`
- GitHub Discussions: Ask questions in the repository discussions

---

## Quick Reference Commands

```bash
# Activate environment
source airsas_env/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Test installation
python test_utilities.py

# Run main script
python makeSasImage.py

# Deactivate environment
deactivate
```

---

**Note**: This installation guide assumes you have basic familiarity with command-line operations. If you encounter issues not covered here, please check the troubleshooting section or open an issue on GitHub.
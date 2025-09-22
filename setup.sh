#!/bin/bash

# AirSAS Python Setup Script
# Automated installation script for the AirSAS Python environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os=$(detect_os)
    
    print_status "Installing system dependencies for $os..."
    
    case $os in
        "linux")
            if command_exists apt-get; then
                print_status "Updating package list..."
                sudo apt-get update
                
                print_status "Installing Python and development tools..."
                sudo apt-get install -y \
                    python3 \
                    python3-pip \
                    python3-venv \
                    python3-dev \
                    build-essential \
                    libhdf5-dev \
                    libffi-dev \
                    libssl-dev \
                    python3-tk
                    
            elif command_exists yum; then
                print_status "Installing packages with yum..."
                sudo yum install -y \
                    python3 \
                    python3-pip \
                    python3-devel \
                    gcc \
                    gcc-c++ \
                    hdf5-devel \
                    libffi-devel \
                    openssl-devel \
                    tkinter
            else
                print_warning "Could not detect package manager. Please install Python 3.8+ manually."
            fi
            ;;
        "macos")
            if command_exists brew; then
                print_status "Installing dependencies with Homebrew..."
                brew install python@3.10 hdf5
            else
                print_warning "Homebrew not found. Please install Python 3.8+ and HDF5 manually."
                print_status "You can install Homebrew from: https://brew.sh"
            fi
            ;;
        "windows")
            print_warning "Windows detected. Please ensure Python 3.8+ is installed."
            print_status "Download from: https://python.org"
            ;;
        *)
            print_warning "Unknown OS. Please install Python 3.8+ manually."
            ;;
    esac
}

# Function to check Python version
check_python_version() {
    print_status "Checking Python version..."
    
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Get Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    print_status "Found Python $PYTHON_VERSION"
    
    # Check if version is 3.8+
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python version is compatible"
    else
        print_error "Python 3.8+ required. Found $PYTHON_VERSION"
        exit 1
    fi
}

# Function to create virtual environment
create_virtual_env() {
    local venv_name=${1:-"airsas_env"}
    
    print_status "Creating virtual environment: $venv_name"
    
    if [ -d "$venv_name" ]; then
        print_warning "Virtual environment $venv_name already exists"
        read -p "Do you want to remove it and create a new one? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$venv_name"
        else
            print_status "Using existing virtual environment"
            return 0
        fi
    fi
    
    $PYTHON_CMD -m venv "$venv_name"
    print_success "Virtual environment created: $venv_name"
}

# Function to activate virtual environment
activate_virtual_env() {
    local venv_name=${1:-"airsas_env"}
    
    print_status "Activating virtual environment..."
    
    if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        source "$venv_name/Scripts/activate"
    else
        source "$venv_name/bin/activate"
    fi
    
    print_success "Virtual environment activated"
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    python -m pip install --upgrade pip
    print_success "Pip upgraded"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install core dependencies first
    print_status "Installing core scientific packages..."
    pip install numpy>=1.21.0
    pip install scipy>=1.7.0
    pip install matplotlib>=3.5.0
    pip install h5py>=3.1.0
    pip install pandas>=1.3.0
    
    # Install from requirements.txt if it exists
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
    fi
    
    print_success "Python dependencies installed"
}

# Function to run tests
run_tests() {
    print_status "Running installation tests..."
    
    # Test basic imports
    python -c "
import numpy as np
import scipy
import matplotlib.pyplot as plt
import h5py
import pandas as pd
print('✓ Core packages imported successfully')
print('  NumPy version:', np.__version__)
print('  SciPy version:', scipy.__version__)
print('  Matplotlib version:', matplotlib.__version__)
print('  H5py version:', h5py.__version__)
print('  Pandas version:', pd.__version__)
"
    
    # Test AirSAS utilities if available
    if [ -f "test_utilities.py" ]; then
        print_status "Running AirSAS utilities test..."
        python test_utilities.py
    else
        print_warning "test_utilities.py not found. Skipping utilities test."
    fi
    
    print_success "All tests passed!"
}

# Function to print usage instructions
print_usage() {
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "To use the AirSAS Python environment:"
    echo ""
    echo "1. Activate the virtual environment:"
    if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        echo "   source airsas_env/Scripts/activate"
    else
        echo "   source airsas_env/bin/activate"
    fi
    echo ""
    echo "2. Run the main script:"
    echo "   python makeSasImage.py"
    echo ""
    echo "3. To deactivate when done:"
    echo "   deactivate"
    echo ""
    echo "📖 See INSTALLATION.md for detailed documentation"
    echo "🐛 Report issues at: https://github.com/dyy5079/In-air_SAS/issues"
    echo ""
}

# Main installation function
main() {
    echo "🚀 AirSAS Python Setup Script"
    echo "=============================="
    echo ""
    
    # Parse command line arguments
    SKIP_SYSTEM_DEPS=false
    VENV_NAME="airsas_env"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-system-deps)
                SKIP_SYSTEM_DEPS=true
                shift
                ;;
            --venv-name)
                VENV_NAME="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-system-deps    Skip system dependency installation"
                echo "  --venv-name NAME      Name for virtual environment (default: airsas_env)"
                echo "  -h, --help           Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    if [ "$SKIP_SYSTEM_DEPS" = false ]; then
        install_system_deps
    else
        print_status "Skipping system dependency installation"
    fi
    
    check_python_version
    create_virtual_env "$VENV_NAME"
    activate_virtual_env "$VENV_NAME"
    upgrade_pip
    install_python_deps
    run_tests
    print_usage
}

# Run main function with all arguments
main "$@"
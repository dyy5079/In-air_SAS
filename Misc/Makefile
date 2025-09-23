# AirSAS Python Development Makefile
# Common tasks for development and maintenance

.PHONY: help install install-dev test test-coverage clean lint format setup docs

# Default target
help:
	@echo "AirSAS Python Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup          - Run automated setup script"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-utils     - Run utilities package tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run code linting (flake8)"
	@echo "  format         - Format code (black + isort)"
	@echo "  type-check     - Run type checking (mypy)"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           - Generate documentation"
	@echo "  docs-serve     - Serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          - Clean up temporary files"
	@echo "  clean-all      - Clean everything including venv"
	@echo "  requirements   - Update requirements files"

# Setup and Installation
setup:
	@echo "Running automated setup..."
	@./setup.sh

install:
	@echo "Installing production dependencies..."
	@pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	@pip install -r requirements-dev.txt

# Testing
test:
	@echo "Running all tests..."
	@python -m pytest tests/ -v

test-coverage:
	@echo "Running tests with coverage..."
	@python -m pytest tests/ --cov=utilities --cov-report=html --cov-report=term

test-utils:
	@echo "Running utilities package tests..."
	@python test_utilities.py

# Code Quality
lint:
	@echo "Running code linting..."
	@flake8 *.py utilities/ --max-line-length=88 --extend-ignore=E203,W503

format:
	@echo "Formatting code..."
	@black *.py utilities/
	@isort *.py utilities/

type-check:
	@echo "Running type checking..."
	@mypy *.py utilities/ --ignore-missing-imports

# Documentation
docs:
	@echo "Generating documentation..."
	@cd docs && make html

docs-serve:
	@echo "Serving documentation locally..."
	@cd docs/_build/html && python -m http.server 8000

# Maintenance
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage

clean-all: clean
	@echo "Cleaning everything..."
	@rm -rf airsas_env/
	@rm -rf .mypy_cache/
	@rm -rf docs/_build/

requirements:
	@echo "Updating requirements files..."
	@pip freeze > requirements-frozen.txt
	@echo "Generated requirements-frozen.txt with exact versions"

# Development workflow targets
dev-setup: install-dev
	@echo "Setting up development environment..."
	@pre-commit install

check: lint type-check test
	@echo "All checks passed!"

ci: install test lint
	@echo "CI pipeline completed!"

# Platform-specific targets
setup-linux:
	@echo "Setting up for Linux..."
	@sudo apt-get update
	@sudo apt-get install -y python3 python3-pip python3-venv python3-dev build-essential libhdf5-dev
	@make install

setup-macos:
	@echo "Setting up for macOS..."
	@brew install python@3.10 hdf5
	@make install

# Docker targets (if needed)
docker-build:
	@echo "Building Docker image..."
	@docker build -t airsas-python .

docker-run:
	@echo "Running Docker container..."
	@docker run -it --rm -v $(PWD):/workspace airsas-python

# Jupyter notebook targets
notebook:
	@echo "Starting Jupyter notebook..."
	@jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Data management targets
sample-data:
	@echo "Setting up sample data structure..."
	@mkdir -p data/scenes
	@mkdir -p "data/characterization data"
	@echo "Sample data directories created"

# Performance testing
benchmark:
	@echo "Running performance benchmarks..."
	@python -m timeit -s "import numpy as np" "np.random.rand(1000, 1000).dot(np.random.rand(1000, 1000))"

# Security check
security-check:
	@echo "Running security checks..."
	@pip install safety
	@safety check

# Update all dependencies
update-deps:
	@echo "Updating all dependencies..."
	@pip install --upgrade pip
	@pip install --upgrade -r requirements.txt

# Environment info
env-info:
	@echo "Environment Information:"
	@echo "======================="
	@python --version
	@pip --version
	@python -c "import numpy; print('NumPy:', numpy.__version__)"
	@python -c "import scipy; print('SciPy:', scipy.__version__)"
	@python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
	@python -c "import h5py; print('H5py:', h5py.__version__)"
	@python -c "import pandas; print('Pandas:', pandas.__version__)"
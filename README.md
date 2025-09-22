# AirSAS Python

Python implementation of Air-coupled Synthetic Aperture Sonar (AirSAS) processing algorithms.

## 🚀 Quick Start

### Automated Installation
```bash
# Clone the repository
git clone https://github.com/dyy5079/In-air_SAS.git
cd In-air_SAS

# Run automated setup (Linux/macOS)
./setup.sh

# Or for Windows
setup.bat
```

### Manual Installation
```bash
# Create virtual environment
python3 -m venv airsas_env
source airsas_env/bin/activate  # Linux/macOS
# airsas_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_utilities.py
```

## 📖 Documentation

- **[Installation Guide](INSTALLATION.md)** - Comprehensive setup instructions
- **[Python Conversion Notes](PYTHON_CONVERSION_README.md)** - Details about MATLAB to Python conversion
- **Configuration** - See `config.ini` for default settings

## 🛠️ Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
make test

# Format code
make format

# See all available commands
make help
```

## 📁 Project Structure

```
In-air_SAS/
├── makeSasImage.py          # Main processing script (converted from MATLAB)
├── utilities/               # Python utility functions package
│   ├── __init__.py
│   ├── packToStruct.py     # Data loading and preprocessing
│   ├── reconstructImage.py # Image reconstruction algorithms
│   ├── plotSasImage.py     # Visualization functions
│   └── ...
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── setup.sh                # Automated setup script (Linux/macOS)
├── setup.bat               # Automated setup script (Windows)
├── test_utilities.py       # Package tests
├── Makefile                # Development commands
└── docs/                   # Documentation
```

## 🔧 Usage

```python
# Basic usage
python makeSasImage.py

# Set data paths in the script or environment
export AIRSAS_DATA_PATH="/path/to/your/data"
```

## 📊 Features

- ✅ Complete Python conversion of MATLAB AirSAS code
- ✅ Robust error handling and path management
- ✅ Package-based architecture with proper imports
- ✅ Comprehensive testing and validation
- ✅ Cross-platform support (Linux, macOS, Windows)
- ✅ Development tools and pre-commit hooks
- ✅ Automated installation scripts

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/dyy5079/In-air_SAS/issues)
- **Questions**: [GitHub Discussions](https://github.com/dyy5079/In-air_SAS/discussions)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
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
├── FeatureIdn_Classification/   # Feature identification and classification scripts
│   ├── env1Idn.py
│   ├── ImgProcessing.py
├── utilities/                   # Python utility functions package
│   ├── __init__.py
│   ├── CFARDetector2D.py
│   ├── freqVecGen.py
│   ├── genLfm.py
│   ├── getAirSpeed.py
│   ├── initStruct.py
│   ├── packToStruct.py
│   ├── plotSasImage.py
│   ├── reconstructImage.py
│   ├── sasColormap.py
├── Misc/                        # Miscellaneous scripts and configs
│   ├── Makefile
│   ├── requirements-dev.txt
│   ├── requirements.txt
│   ├── setup.bat
│   ├── setup.sh
├── README/                      # Documentation
│   ├── INSTALLATION.md
│   ├── PYTHON_CONVERSION_README.md
├── config.ini                   # Default configuration
├── makeSasImage.py              # Main processing script (converted from MATLAB)
├── cropTarget.py                # Cropping utility
├── kSpaceCrop.py                # k-space cropping utility
├── saveh5.py                    # HDF5 saving utility
├── README.md                    # Project overview
├── .gitignore                   # Git ignore rules
```

**Note:**
- The `data/` and `outputs/` directories are ignored by git (see `.gitignore`). They will not be present in the GitHub repository. You must create these locally and add your own data and results.
- The `venv/` and `__pycache__/` directories are also ignored and should be created locally as needed.
- Data used for the FeatureIdn_Classification can be found at 
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
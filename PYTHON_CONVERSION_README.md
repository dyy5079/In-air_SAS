# Python Conversion of makeSasImage.m

This directory contains the Python translation of the MATLAB AirSAS processing scripts.

## Files

- `makeSasImage.py` - Main script converted from MATLAB `makeSasImage.m`
- `utilities/` - Directory containing Python versions of all MATLAB utility functions

## Key Differences from MATLAB Version

### 1. Error Handling
The Python version includes robust error handling for:
- Missing dependencies (numpy, h5py, matplotlib)
- File not found errors
- Different HDF5 file structures
- Import path issues

### 2. Package Structure
The utilities are now properly organized as a Python package with `__init__.py`:
```python
# Clean package-level imports
from utilities import (
    packToStruct, 
    reconstruct_image, 
    plotSasImage, 
    sas_colormap
)
```

The utilities package includes:
- Proper relative imports (e.g., `from .sasColormap import sas_colormap`)
- Package-level exports in `__init__.py`
- Error handling for missing dependencies
    except ImportError:
        utilities_available = False
```

### 3. Graceful Degradation
- Basic image loading works even if advanced utilities are not available
- Falls back to standard colormaps if custom SAS colormap is unavailable
- Provides informative error messages

### 4. Bug Fixes
- Fixed HDF5 path for imaginary component: `/ch{loadCh}/img_im` (was incorrectly `/ch{loadCh}/img_re` in original)
- Added flexible HDF5 path handling for different file structures

## Testing the Package

Run the test script to verify everything is working:
```bash
python test_utilities.py
```

This will test:
- Package imports work correctly
- Basic functionality of key utilities
- Package structure integrity

## Usage

### Basic Usage
```python
# Set your data paths
dataFolder = '/path/to/your/data'  # Contains both 'scenes' and 'characterization data' folders
filename = 't1e4_01.h5'           # Your HDF5 data file

# Run the script
python makeSasImage.py
```

### Required Dependencies
```bash
pip install numpy h5py matplotlib scipy pandas
```

### Data Structure Expected
```
dataFolder/
├── scenes/
│   └── filename.h5  # HDF5 file with SAS data
└── characterization data/
    ├── acquistionParams.csv
    └── sensorCoordinates.csv
```

### HDF5 File Structure Expected
```
filename.h5:
├── /ch1/
│   ├── img_re  # Real part of reconstructed image
│   ├── img_im  # Imaginary part of reconstructed image
│   └── ts      # Time series data
├── /ch2/...
├── /ch3/...
├── /ch4/...
└── /na/
    ├── xVec        # Along-track coordinates
    ├── yVec        # Cross-track coordinates
    ├── temperature # Temperature measurements
    ├── humidity    # Humidity measurements
    └── position    # Position data
```

## Function Equivalents

| MATLAB Function | Python Function | Location |
|----------------|-----------------|----------|
| `packToStruct` | `packToStruct` | `utilities/packToStruct.py` |
| `reconstructImage` | `reconstruct_image` | `utilities/reconstructImage.py` |
| `plotSasImage` | `plotSasImage` | `utilities/plotSasImage.py` |
| `sasColormap` | `sas_colormap` | `utilities/sasColormap.py` |

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all utility files are in the `utilities/` directory or current working directory.

2. **File Not Found**: Check that `dataFolder` and `filename` variables are correctly set to point to your data.

3. **HDF5 Structure Mismatch**: The script attempts to handle different HDF5 structures automatically, but you may need to adjust paths if your file structure differs.

4. **Missing Dependencies**: Install required packages:
   ```bash
   pip install numpy h5py matplotlib scipy pandas
   ```

### Debug Mode
The script prints available HDF5 groups and datasets to help diagnose file structure issues:
```
Available groups in file: ['ch1', 'ch2', 'ch3', 'ch4', 'na']
```

## Migration Notes

### Key Changes from MATLAB:
1. **Array Indexing**: Python uses 0-based indexing vs MATLAB's 1-based
2. **Complex Numbers**: `1j` in Python vs `1i` in MATLAB  
3. **Array Operations**: Uses NumPy for array operations
4. **File I/O**: Uses h5py for HDF5 files vs MATLAB's h5read
5. **Plotting**: Uses matplotlib vs MATLAB's built-in plotting

### Performance Considerations:
- Python version may be slower for large datasets due to interpreted nature
- Consider using NumPy's optimized functions for better performance
- Use `np.asarray()` for type consistency when needed

## Example Output

When successful, you should see:
```
Loading and plotting basic SAS image...
Available groups in file: ['ch1', 'ch2', 'ch3', 'ch4', 'na']
Successfully loaded image with shape: (800, 1667)
Backprojection of Channel 1 Complete
Backprojection of Channel 2 Complete
Backprojection of Channel 3 Complete
Backprojection of Channel 4 Complete
```

Plus matplotlib windows showing the SAS imagery.
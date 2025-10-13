import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
from utilities import packToStruct, reconstructImage, plotSasImage, sasColormap
from matplotlib.colors import ListedColormap
import glob

# Specify the path and filename for the data before execution
folder = ''  # path to folder containing both \scenes and \characterization data
if folder == '':
    print("Error: 'folder' variable is not set. Please specify the path to the data folder.")
    exit()

filename = 't*.h5'
pattern = os.path.join(folder, 'scenes', filename)
dataList = glob.glob(pattern)  # this returns a list of matching files

# Check if any files were found
if not dataList:
    print(f"Error: No files matching pattern '{filename}' found in {os.path.join(folder, 'scenes')}")
    print("Please ensure data files exist at the specified location.")
    exit()

# for debugging, print the found files
print(f"Found {len(dataList)} files:")  # new line
for file in dataList:  # new line
    print(f"  {os.path.basename(file)}")  # new line

excludeList = glob.glob(os.path.join(folder, 'scenes', 't0*.h5'))  # exclude t0 files which have no targets

dataList = [file for file in dataList if file not in excludeList]  # subtract excludeList from dataList

# for debugging, print the remaining files after exclusion
print(f"After exclusions, {len(dataList)} files remain:")  # new line
for file in dataList:  # new line
    print(f"  {os.path.basename(file)}")  # new line


p = 1.0e-7  # probability of false alarm for CFAR detector, adjust as needed
nGuard = 6  # number of guard cells on each side of the cell under test
nTrain = 10  # number of training cells on each side of the cell under test

#detector = CFARDetector 2D

chipLx = 0.5
chipLy = 0.5

#nominal positions of the targets (x,y,z)
targetPos = np.array([
    [-1.125, .866, 0],
    [-.375, .866, 0],
    [.375, .866, 0],
    [1.125, .866, 0],
    [-.75, 1.616, 0],
    [0, 1.616, 0],
    [.75, 1.616, 0]]) + np.array([2.75, 0, 0])

lBox = 12 * 0.0254
targList = {'Solid Sphere', 'Hollow Sphere', 'O', 'Q'}
envList = {'Free Field', 'Flat Interface', 'Rough Interface', 'Partially Buried'}
targDimY = np.array([4, 4, 8, 8]) * 0.0254  # nominal dimensions of each target in Y
targDimX = np.array([4, 4, 8, 9.8532]) * 0.0254  # nominal dimensions of each target in X

# Preallocate results matrices
centerX = np.full((len(dataList), 7), np.nan)  
centerY = np.full((len(dataList), 7), np.nan)  
dimX = np.full((len(dataList), 7), np.nan)     
dimY = np.full((len(dataList), 7), np.nan)     

for m in range(len(dataList)):
    filename = dataList[m]
    print(f"Processing file {m+1}/{len(dataList)}: {os.path.basename(filename)}")
    dPath = os.path.join(folder, 'scenes', filename)

    loadCh = 1

    try:
        with h5py.File(dPath, 'r') as f:
            sasImg = f[f'/ch{loadCh}/img_re'][:] + 1j * f[f'/ch{loadCh}/img_im'][:]  # complex-valued SAS image
            xVec = f['/na/xVec'][:]  # vector of pixels coordinates in the along-track direction, m
            yVec = f['/na/yVec'][:]  # vector of pixels coordinates in the cross-track direction, m
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        exit()

    # the reconstructed imagery is spatially oversampled.  The images will be
    # decimated to be critically sampled prior to detection.
    ratio = 3
    xVec = xVec[::ratio]  # take every 3rd element
    yVec = yVec[::ratio]  # take every 3rd element  
    sasImg = sasImg[::ratio, ::ratio]  # downsample the 2D image by factor of 3

    # Apply range normalization to the SAS image to account for the decaying
    # intensity with increasing range from the sensor. This step is crucial for
    rNorm = np.tile(yVec, (len(xVec), 1))  # new line - equivalent to repmat(yVec, numel(xVec), 1)
    sasImg = sasImg * rNorm  # new line - element-wise multiplication for range normalization    



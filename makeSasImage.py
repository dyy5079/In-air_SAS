"""
Script to load AirSAS data from .h5 file, process it, and plot the results

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
from utilities import packToStruct, reconstructImage, plotSasImage, sasColormap
from matplotlib.colors import ListedColormap
# Setup the paths to the data file and code repository for processing and analysis

# Specify the path to the code repository
basePath = '/home/In-air_SAS'  # path to the In-air_SAS repository
if basePath == '':
    basePath = os.getcwd()
sys.path.append(os.path.join(basePath, 'utilities'))    #add the utilities folder 

# Specify the path and filename for the data before execution
dataFolder = ''  # path to folder containing both \scenes and \characterization data
if dataFolder == '':
    dataFolder = os.path.join(basePath, 'data')  # default path to data folder
filename = 't4e2_06.h5'  # specific .h5 data file to load, t1e4_01.h5 will reproduce Fig 10b
dPath = os.path.join(dataFolder, 'scenes', filename)

# Check if file exists
if not os.path.exists(dPath):
    print(f"Error: File not found at {dPath}")
    print("Please ensure the data file exists at the specified location.")
    exit()

# load and plot a SAS image
loadCh = 4  # data channel to load

try:
    with h5py.File(dPath, 'r') as f:
        tsRaw = f[f'/ch{loadCh}/img_re'][:] + 1j * f[f'/ch{loadCh}/img_im'][:]  # complex-valued SAS image
        xVec = f['/na/xVec'][:]  # vector of pixels coordinates in the along-track direction, m
        yVec = f['/na/yVec'][:]  # vector of pixels coordinates in the cross-track direction, m
except Exception as e:
    print(f"Error reading HDF5 file: {e}")
    exit()

# Create output directory for saving plots
output_dir = os.path.join(basePath, 'output_plots')
os.makedirs(output_dir, exist_ok=True)

# plot the magnitude of the image
plt.figure(1, figsize=(12, 4))  # width=12 inches, height=4 inches
plt.imshow(20 * np.log10(np.abs(tsRaw) + 1e-12),       # adding 1e-12 to avoid log of zero
           extent=[xVec[0], xVec[-1], yVec[0], yVec[-1]], 
           aspect='auto', 
           origin='lower',
           cmap=ListedColormap(sasColormap()))
plt.xlabel('Along-track (m)')
plt.ylabel('Cross-track (m)')
plt.clim([0, 30])
h = plt.colorbar()
h.set_label('Amplitude (dB re: 1V @ 1m)')

# Add more tick marks for better readability
plt.xticks(np.arange(xVec[0].item(), xVec[-1].item(), 0.5))  # X-axis ticks every 0.5 meters
plt.yticks(np.arange(yVec[0].item(), yVec[-1].item(), 0.2))  # Y-axis ticks every 0.2 meters

#plt.gca().invert_yaxis()  # 0 at top for y-axis
#plt.gca().invert_xaxis()  # 0 at right for x-axis

plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_ch{loadCh}.png'), dpi=300, bbox_inches='tight')
#plt.show()

print("Loading and plotting basic SAS image...")

print(f"Successfully loaded image with shape: {tsRaw.shape}")

# Load the complete set of data and pre-process the time series
# Here the raw acoustic data, along with all of the non-acoustic parameters
# are loaded and pre-processed
chanSelect = [1, 2, 3, 4]  # select which of the receiver channels to load
cSelect = 0  # flag for which sound speed model to use. 0=temp only, 1=temp+humidity
A = packToStruct(dataFolder, filename, chanSelect, cSelect)  # load the data, and pre-process the time series

# Reconstruct an image from the data using backprojection

# Define parameters for reconstructing the imagery
cross_track = [0.1, 2.4]  # bounds of the imagery in the cross-track direction, m
dy = 3e-3  # pixel size in the cross-track direction, m
along_track = [0.3, 5.3]  # bounds of the imagery in the along-track direction, m
dx = 3e-3  # pixel size in the along-track direction, m
img_plane = 0.6032  # elevation of imaging plane, m

resampling_ratio = 10  # upsampling ratio prior to nearest neighbor interpolation
fov = 120  # field of view from the transmitter to the pixel to include in the integration, degrees

# Define plotting parameters for displaying the imagery
normFlag = 0  # flag to apply 30*log10(r) range normalization to the imagery 
              # (1 = normalization on, 0 = normalization off)
dynamicRange = 35  # dynamic range to display in the image, dB

# Reconstruct imagery one channel (one microphone) at a time
backprojectionImg = []
#for m in range(len(chanSelect)):
for m in range(1):  # for testing, only reconstruct channel 1
    # pass the image reconstruction parameters to the data structure
    A[m].Results.Bp.xVect = np.arange(along_track[0], along_track[1] + dx, dx)
    A[m].Results.Bp.yVect = np.arange(cross_track[0], cross_track[1] + dy, dy)
    A[m].Results.Bp.zPlane = img_plane
    A[m].Results.Bp.fov = fov
    
    # reconstruct the imagery
    backprojectionImg.append(reconstructImage(A[m], resampling_ratio, img_plane, fov))
    print(f'Backprojection of Channel {chanSelect[m]} Complete')
    
    # plot the reconstructed imagery
    plt.figure(figsize=(12, 4))  # width=12 inches, height=4 inches
    plotSasImage(backprojectionImg[m], dynamicRange, normFlag, output_dir, filename, chanSelect, m)
    plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_ch{chanSelect[m]}_Backprojection_ch{m}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
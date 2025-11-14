"""
Script to load AirSAS data from .h5 file, process it, and plot the results

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
import glob
from utilities import packToStruct, reconstructImage, plotSasImage, sasColormap
from cropTarget import cropTarget
from kSpaceCrop import kSpaceCrop
from saveh5 import saveh5

from matplotlib.colors import ListedColormap
# Setup the paths to the data file and code repository for processing and analysis

# Specify the path to the code repository
basePath = ''  # path to the In-air_SAS repository
if basePath == '':
    basePath = os.getcwd()
sys.path.append(os.path.join(basePath, 'utilities'))    #add the utilities folder 

# Specify the path and filename for the data before execution
dataFolder = ''  # path to folder containing both \scenes and \characterization data
if dataFolder == '':
    dataFolder = os.path.join(basePath, 'data')  # default path to data folder

# Get all .h5 files in the scenes folder
scenes_path = os.path.join(dataFolder, 'scenes')
h5_files = glob.glob(os.path.join(scenes_path, 't4e*.h5'))

if not h5_files:
    print(f"Error: No .h5 files found in {scenes_path}")
    exit()

print(f"Found {len(h5_files)} .h5 files to process:")
for f in h5_files:
    print(f"  - {os.path.basename(f)}")
print()

# Create output directory for saving plots
output_dir = os.path.join(basePath, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Process each file
for dPath in h5_files:
    filename = os.path.basename(dPath)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")
    
    # load and plot a SAS image
    loadCh = 1  # data channel to load

    try:
        with h5py.File(dPath, 'r') as f:
            tsRaw = f[f'/ch{loadCh}/img_re'][:] + 1j * f[f'/ch{loadCh}/img_im'][:]  # complex-valued SAS image
            xVec = f['/na/xVec'][:]  # vector of pixels coordinates in the along-track direction, m
            yVec = f['/na/yVec'][:]  # vector of pixels coordinates in the cross-track direction, m
    except Exception as e:
        print(f"Error reading HDF5 file {filename}: {e}")
        continue

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
    normFlag = 1  # flag to apply 30*log10(r) range normalization to the imagery 
                  # (1 = normalization on, 0 = normalization off)
    dynamicRange = 35  # dynamic range to display in the image, dB

    # Reconstruct imagery one channel (one microphone) at a time
    backprojectionImg = []
    #for m in range(len(chanSelect)):
    for m in range(1):  # Only reconstruct channel 1
        # pass the image reconstruction parameters to the data structure
        A[m].Results.Bp.xVect = np.arange(along_track[0], along_track[1] + dx, dx)
        A[m].Results.Bp.yVect = np.arange(cross_track[0], cross_track[1] + dy, dy)
        A[m].Results.Bp.zPlane = img_plane
        A[m].Results.Bp.fov = fov
        
        # reconstruct the imagery
        backprojectionImg.append(reconstructImage(A[m], resampling_ratio, img_plane, fov))
        print(f'Backprojection of Channel {chanSelect[m]} Complete')
        
        # Save the reconstructed image to an HDF5 file
        h5outputdir = os.path.join(basePath, 'outputs','backprojection_h5')
        os.makedirs(h5outputdir, exist_ok=True)
        saveh5(backprojectionImg[m], filename=filename, output_dir=h5outputdir, channel=chanSelect[m])

        # plot the reconstructed imagery
        bpoutputdir = os.path.join(basePath, 'outputs','backprojection')
        os.makedirs(bpoutputdir, exist_ok=True)
        plt.figure(figsize=(12, 4))  # width=12 inches, height=4 inches
        plotSasImage(backprojectionImg[m], dynamicRange, normFlag, filename, chanSelect, m)
        plt.savefig(os.path.join(bpoutputdir, f'{filename[:-3]}_ch{chanSelect[m]}_Backprojection_ch{m}.png'), dpi=300, bbox_inches='tight')
        plt.close('all')
        chipoutputdir = os.path.join(basePath, 'outputs','chip')
        kchipoutputdir = os.path.join(basePath, 'outputs','kchip')
        os.makedirs(chipoutputdir, exist_ok=True)
        os.makedirs(kchipoutputdir, exist_ok=True)
        cropTarget(backprojectionImg[m], dynamicRange=dynamicRange, normFlag=normFlag, filename=filename, plot=True, output_dir=chipoutputdir, channel=chanSelect[m])
        kSpaceCrop(backprojectionImg[m], dynamicRange=dynamicRange, normFlag=False, filename=filename, plot=True, output_dir=kchipoutputdir, channel=chanSelect[m])

    print(f"Completed processing: {filename}")
    # Clear all matplotlib figures and free memory
    plt.clf()  # Clear current figure
    plt.cla()  # Clear current axes
    plt.close('all')

    # Force garbage collection
    import gc
    gc.collect()

print(f"\n{'='*60}")
print(f"All files processed successfully!")
print(f"{'='*60}")
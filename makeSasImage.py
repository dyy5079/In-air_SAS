"""
Script to load AirSAS data from .h5 file, process it, and plot the results

Python translation of makeSasImage.m
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os

# Setup the paths to the data file and code repository for processing and analysis

# Specify the path to the code repository
basePath = ''
sys.path.append(os.path.join(basePath, 'utilities'))

# Import utilities with error handling
try:
    from utilities import packToStruct, reconstruct_image, plotSasImage, sas_colormap
    utilities_available = True
except ImportError as e:
    print(f"Warning: Could not import utilities: {e}")
    utilities_available = False

try:
    from matplotlib.colors import ListedColormap
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("Warning: matplotlib not available. Plotting functionality limited.")

# Specify the path and filename for the data before execution
dataFolder = ''  # path to folder containing both \scenes and \characterization data
filename = 't1e4_01.h5'  # specific .h5 data file to load, t1e4_01.h5 will reproduce Fig 10b
dPath = os.path.join(dataFolder, 'scenes', filename)

def load_and_plot_basic_image():
    """Load and plot a basic SAS image from HDF5 file"""
    # load and plot a SAS image
    loadCh = 1  # data channel to load
    
    try:
        with h5py.File(dPath, 'r') as f:
            # Print available datasets for debugging
            print(f"Available groups in file: {list(f.keys())}")
            
            # Try different possible paths for image data
            try:
                img_real = f[f'/ch{loadCh}/img_re'][:]
                img_imag = f[f'/ch{loadCh}/img_im'][:]  # Fixed: should be img_im, not img_re
            except KeyError:
                try:
                    img_real = f[f'ch{loadCh}/img_re'][:]
                    img_imag = f[f'ch{loadCh}/img_im'][:]
                except KeyError:
                    print(f"Could not find image data for channel {loadCh}")
                    return None
            
            sasImg = img_real + 1j * img_imag  # complex-valued SAS image
            
            # Try different possible paths for coordinate vectors
            try:
                xVec = f['/na/xVec'][:]  # vector of pixels coordinates in the along-track direction, m
                yVec = f['/na/yVec'][:]  # vector of pixels coordinates in the cross-track direction, m
            except KeyError:
                try:
                    xVec = f['na/xVec'][:]
                    yVec = f['na/yVec'][:]
                except KeyError:
                    print("Could not find coordinate vectors, using indices")
                    yVec = np.arange(sasImg.shape[0])
                    xVec = np.arange(sasImg.shape[1])
        
        # plot the magnitude of the image
        if matplotlib_available:
            plt.figure(1)
            
            # Create colormap if available
            if utilities_available:
                try:
                    cmap = ListedColormap(sas_colormap())
                except:
                    cmap = 'jet'  # fallback colormap
            else:
                cmap = 'jet'
            
            plt.imshow(20 * np.log10(np.abs(sasImg)), 
                       extent=[xVec[0], xVec[-1], yVec[-1], yVec[0]], 
                       aspect='auto', 
                       vmin=0, vmax=30, 
                       cmap=cmap)
            plt.xlabel('Along-track (m)')
            plt.ylabel('Cross-track (m)')
            plt.colorbar()
            plt.title('SAS Image Magnitude (dB)')
            plt.show()
        
        return sasImg, xVec, yVec
        
    except FileNotFoundError:
        print(f"Error: Could not find file {dPath}")
        print("Please ensure dataFolder and filename are correctly set.")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def main():
    """Main function to load and plot SAS image data"""
    
    print("Loading and plotting basic SAS image...")
    result = load_and_plot_basic_image()
    
    if result is None:
        print("Could not load basic image. Exiting.")
        return
    
    sasImg, xVec, yVec = result
    print(f"Successfully loaded image with shape: {sasImg.shape}")
    
    # Only proceed with advanced processing if utilities are available
    if not utilities_available:
        print("Advanced processing functions not available. Only basic image loading was performed.")
        return
    
    # Load the complete set of data and pre-process the time series
    # Here the raw acoustic data, along with all of the non-acoustic parameters
    # are loaded and pre-processed
    try:
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
        for m in range(len(chanSelect)):
            # pass the image reconstruction parameters to the data structure
            A[m].Results.Bp.xVect = np.arange(along_track[0], along_track[1] + dx, dx)
            A[m].Results.Bp.yVect = np.arange(cross_track[0], cross_track[1] + dy, dy)
            A[m].Results.Bp.zPlane = img_plane
            A[m].Results.Bp.fov = fov
            
            # reconstruct the imagery
            A[m] = reconstruct_image(A[m], resampling_ratio, img_plane, fov)
            print(f'Backprojection of Channel {chanSelect[m]} Complete')
            
            # plot the reconstructed imagery
            if matplotlib_available:
                plt.figure()
                plotSasImage(A[m], dynamicRange, normFlag)
                plt.show()
    
    except Exception as e:
        print(f"Error in advanced processing: {e}")
        print("Basic image loading completed successfully.")


if __name__ == "__main__":
    main()
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.colors import ListedColormap
from utilities import sasColormap

def cropTarget(sasImg, xVec, yVec, plot=True):
    """
    Crop target chips from the SAS image based on the target positions.

    Parameters:
    sasImg : 2D numpy array
        The complex-valued SAS image.
    xVec : 1D numpy array
        The vector of pixel coordinates in the along-track direction (m).
    yVec : 1D numpy array
        The vector of pixel coordinates in the cross-track direction (m).
    plot : bool, optional
        Whether to plot the chips (default: True).

    Returns:
    chips : list of 2D numpy arrays
        The cropped target chips from the SAS image.
    """
    # Defined chip size from the cfarDetector.m
    chipLx = 0.5
    chipLy = 0.5

    # Nominal positions of the targets (x,y,z) from cfarDetector.m
    targetPos = np.array([
        [-1.125, .866, 0],
        [-.375, .866, 0],
        [.375, .866, 0],
        [1.125, .866, 0],
        [-.75, 1.616, 0],
        [0, 1.616, 0],
        [.75, 1.616, 0]]) + np.array([2.75, 0, 0])

    # Calculate half chip sizes in pixels
    dx = xVec[1] - xVec[0]
    dy = yVec[1] - yVec[0]
    halfChipLx = int((chipLx / 2) / dx)
    halfChipLy = int((chipLy / 2) / dy)

    # Store all chips
    chips = []
    
    # Process each target
    for i, pos in enumerate(targetPos):
        # Find the indices corresponding to the target position
        xIdx = np.argmin(np.abs(xVec - pos[0]))
        yIdx = np.argmin(np.abs(yVec - pos[1]))

        # Crop the chip from the SAS image
        chip = sasImg[yIdx - halfChipLy:yIdx + halfChipLy, 
                      xIdx - halfChipLx:xIdx + halfChipLx]
        chips.append(chip)

    # Plot the chips if requested
    if plot:
        # Create individual square plots for each chip
        for i, (chip, pos) in enumerate(zip(chips, targetPos)):
            # Create a new figure for each chip with square aspect ratio
            plt.figure(i + 1)
            
            # Plot magnitude in dB
            chipNormalized = 20 * np.log10(np.abs(chip) + 1e-10)
            
            # Create extent for proper axis labels
            
            
            plt.imshow(chipNormalized, 
                          aspect='equal',  # Square aspect ratio
                          cmap=ListedColormap(sasColormap()),
                          extent = [-chipLx/2, chipLx/2, -chipLy/2, chipLy/2],
                          origin='lower')
            plt.title(f'Target {i+1} at ({pos[0]:.2f}, {pos[1]:.2f}) m', fontsize=14)
            plt.xlabel('Along-track (m)', fontsize=12)
            plt.ylabel('Cross-track (m)', fontsize=12)
            plt.clim([0, 30])
            h = plt.colorbar()
            h.set_label('Amplitude (dB re: 1V @ 1m)')
            
            #plt.tight_layout()
            
    plt.show()
    return chips
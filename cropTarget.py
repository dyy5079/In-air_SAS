import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
import re
from matplotlib.colors import ListedColormap
from utilities import sasColormap


def fileAttribute(filename):
    # Pattern: t<digit(s)>e<digit(s)>_<digit(s)>
    pattern = r't(\d+)e(\d+)_(\d+).h5'
    match = re.match(pattern, filename)

    if match:
        t = int(match.group(1))
        e = int(match.group(2))
        trial = int(match.group(3))
        
        # Customize these descriptions based on your experiment
        target = {
            0: "No Targets",
            1: "Solid Sphere",
            2: "Hollow Sphere",  # Customize this
            3: "Letter O",
            4: "Letter Q",
        }
        
        env = {
            1: "Free Field",  # Customize this
            2: "Flat Interface",
            3: "Rough Interface",
            4: "Partially Buried in Rough Interface",
        }
        
        targetDC = target.get(t, f"Target Type {t}")
        envDC = env.get(e, f"Experiment {e}")

        return {
            'target': targetDC,
            'env': envDC,
            'trial': trial,
        }
    else:
        return None

def cropTarget(sasImg, xVec, yVec, plot=True, filename=None):
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
    filename : str, optional
        The filename to decode for image information (e.g., 't2e1_06.h5')

    Returns:
    chips : list of 2D numpy arrays
        The cropped target chips from the SAS image.
    """
    
    # Decode filename if provided
    file = None
    if filename:
        file = fileAttribute(filename)
        if file:
            print(f"File: {filename}")
            print(f"  Target Type: {file['target']}")
            print(f"  Environment: {file['env']}")
            print(f"  Trial: {file['trial']}")

    # Defined chip size from the cfarDetector.m
    chipLx = 0.5
    chipLy = 0.5

    # Nominal positions of the targets (x,y,z) from cfarDetector.m
    targetPos = np.array([
        [-.75, 1.616, 0],
        [0, 1.616, 0],
        [.75, 1.616, 0],
        [-1.125, .866, 0],
        [-.375, .866, 0],
        [.375, .866, 0],
        [1.125, .866, 0]
        ]) + np.array([2.75, 0, 0])

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
            
            # Create title with filename info if available
            title = f'Target {i+1} at ({pos[0]:.2f}, {pos[1]:.2f}) m'
            if file:
                title += f"\n{file['target']} - {file['env']} (Trial {file['trial']})"
            plt.title(title, fontsize=14)
            
            plt.xlabel('Along-track (m)', fontsize=12)
            plt.ylabel('Cross-track (m)', fontsize=12)
            plt.clim([0, 30])
            h = plt.colorbar()
            h.set_label('Amplitude (dB re: 1V @ 1m)')
            
            #plt.tight_layout()
            
    plt.show()
    return chips
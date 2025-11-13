import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import ndimage
import sys
import os
from matplotlib.colors import ListedColormap
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities import sasColormap

def read_h5(filepath):
    """
    Reads the custom h5 file and returns its datasets as a dictionary.
    Expects datasets:
      - 'target': bytes
      - 'env': bytes
      - 'trial': int32
      - 'tsRC': complex64 array
    """
    with h5py.File(filepath, 'r') as file:
        target = file['target'][()]
        env = file['env'][()]
        trial = file['trial'][()].astype(np.int32)
        chip = file['chip'][()].astype(np.complex64)
        # decode bytes if needed
        if isinstance(target, bytes):
            target = target.decode('utf-8')
        if isinstance(env, bytes):
            env = env.decode('utf-8')
        return {
            'target': target,
            'env': env,
            'trial': trial,
            'chip': chip
        }


def findCenter(img, debug=False, threshold=90, cutArea=1.0, MM=5):
    """
    Find center of an 'O' (ring) in a 2D data matrix using thresholding and vertical bounds.
    Parameters:
      img: 2D numpy array (can be complex)
      debug: if True, show diagnostic plots
    Returns:
      (cx, cy) in pixel coordinates (float) as (col, row), or None if not found.

    Steps:
      1. Convert input to normalized dB magnitude image.
      2. Threshold the image at the provided percentile to create a binary mask.
      3. Clean the mask using morphological opening, closing, and hole filling.
      4. For each x-column, find the highest and lowest y pixel where mask==1.
      5. Compute the center as the mean x, and mean of (y_highest + y_lowest)/2 for all valid columns.
      6. If debug=True, plot the image, mask, detected center, and an estimated O circle.
    """
    sasImg = 20 * np.log10(np.abs(img) + 1e-12)
    sasImg = sasImg - sasImg.min()
    if sasImg.max() != 0:
        sasImg = sasImg / sasImg.max()

    height, width = sasImg.shape
    extent = [0, 50, 0, 50]  # x: 0-50 cm, y: 0-50 cm

    # Plot the normalized dB magnitude image (will add center overlay later)
    if debug:
        plt.figure()
        plt.imshow(sasImg, cmap=ListedColormap(sasColormap()), aspect='equal', extent=extent, origin='lower')
        plt.title('Normalized dB Magnitude (A_dB)')
        plt.colorbar(label='Normalized dB')
        plt.xlabel('Along-track (cm)', fontsize=10)
        plt.ylabel('Cross-track (cm)', fontsize=10)
        plt.show()

    # Threshold at 90th percentile to create binary mask
    if threshold is None:
        print("Threshold value not provided. Aborting.")
        return None
    th = np.percentile(sasImg, threshold)
    mask = sasImg > th
    if debug:
        plt.figure()
        plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        plt.title(f'Binary Mask ({threshold}th Percentile Threshold)')
        plt.xlabel('Along-track (cm)', fontsize=10)
        plt.ylabel('Cross-track (cm)', fontsize=10)
        plt.show()

    mask = ndimage.binary_opening(mask, structure=np.ones((MM, MM)))
    if debug:
        plt.figure()
        plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        plt.title(f'Binary Mask after binary_opening({threshold}th Percentile Threshold)')
        plt.xlabel('Along-track (cm)', fontsize=10)
        plt.ylabel('Cross-track (cm)', fontsize=10)
        plt.show()

    mask = ndimage.binary_closing(mask, structure=np.ones((MM, MM)))
    if debug:
        plt.figure()
        plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        plt.title(f'Binary Mask after binary_closing({threshold}th Percentile Threshold)')
        plt.xlabel('Along-track (cm)', fontsize=10)
        plt.ylabel('Cross-track (cm)', fontsize=10)
        plt.show()

    # Use center of mass for x-coordinate and vertical bounds for y-coordinate
    if mask.sum() > 10:
        cy_com, cx_com = ndimage.center_of_mass(mask)
        y_coords, x_coords = np.where(mask == 1)
        
        if len(y_coords) > 0:
            y_highest = np.min(y_coords)  # highest y (smallest index, since y=0 is at top)
            y_lowest = np.max(y_coords)   # lowest y (largest index)
            
            # If highest y is 0, find the next highest y pixel
            if y_highest == 0 and len(np.unique(y_coords)) > 1:
                unique_y = np.unique(y_coords)
                y_highest = unique_y[1]
            
            # Center uses x from center of mass and y halfway between highest and lowest
            cx = cx_com
            cy = (y_highest + y_lowest) / 2
            center = (cx, cy)
        else:
            center = None

        if debug:
            radius_cm = cutArea * (20.3 / 2)
            
            # Plot dB magnitude with detected center overlay
            plt.figure()
            plt.imshow(sasImg, cmap=ListedColormap(sasColormap()), aspect='equal', extent=extent, origin='lower')
            plt.plot(cx * (50/width), cy * (50/height), 'rx', markersize=10, label='Detected Center')
            circle_overlay = plt.Circle((cx * (50/width), cy * (50/height)), radius_cm, color='cyan', fill=False, linewidth=2, label='Assumed O')
            plt.gca().add_patch(circle_overlay)
            plt.title('Normalized dB Magnitude with Detected Center Overlay')
            plt.colorbar(label='Normalized dB')
            plt.xlabel('Along-track (cm)', fontsize=10)
            plt.ylabel('Cross-track (cm)', fontsize=10)
            plt.legend()
            plt.show()
        return mask, center
    return mask, None
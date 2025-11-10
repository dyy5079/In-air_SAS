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


def findCenter(img, debug=False, threshold=90):
    """
    Find center of an 'O' (ring) in a 2D data matrix using thresholding and centroid.
    Parameters:
      img: 2D numpy array (can be complex)
      debug: if True, show diagnostic plots
    Returns:
      (cx, cy) in pixel coordinates (float) as (col, row), or None if not found.

    Steps:
      1. Convert input to normalized dB magnitude image.
      2. Threshold the image at the provided percentile to create a binary mask.
      3. Clean the mask using morphological opening, closing, and hole filling.
      4. If a sufficient region is found, compute its centroid as the O center.
      5. If debug=True, plot the image, mask, detected center, and an estimated O circle.
    """
    sasImg = 20 * np.log10(np.abs(img) + 1e-12)
    sasImg = sasImg - sasImg.min()
    if sasImg.max() != 0:
        sasImg = sasImg / sasImg.max()

    # Set up axis scaling for 50cm x 50cm image
    height, width = sasImg.shape
    extent = [0, 50, 0, 50]  # x: 0-50 cm, y: 0-50 cm

    # Plot the normalized dB magnitude image
    if debug:
        plt.figure()
        plt.imshow(sasImg, cmap=ListedColormap(sasColormap()), aspect='equal', extent=extent, origin='lower')
        plt.title('Normalized dB Magnitude (A_dB) t3e2_06')
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

    mask = ndimage.binary_opening(mask, structure=np.ones((5,5)))
    if debug:
        plt.figure()
        plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        plt.title(f'Binary Mask after binary_opening({threshold}th Percentile Threshold)')
        plt.xlabel('Along-track (cm)', fontsize=10)
        plt.ylabel('Cross-track (cm)', fontsize=10)
        plt.show()

    mask = ndimage.binary_closing(mask, structure=np.ones((5,5)))
    if debug:
        plt.figure()
        plt.imshow(mask, cmap='gray', extent=extent, origin='lower', aspect='equal')
        plt.title(f'Binary Mask after binary_closing({threshold}th Percentile Threshold)')
        plt.xlabel('Along-track (cm)', fontsize=10)
        plt.ylabel('Cross-track (cm)', fontsize=10)
        plt.show()

    if mask.sum() > 10:
        cy, cx = ndimage.center_of_mass(mask)
        if np.isfinite(cx) and np.isfinite(cy):
            if debug:
                # Estimate radius in cm (diameter is 20.3 cm)
                radius_cm = 1.15 * (20.3 / 2)

                fig, ax = plt.subplots()
                im = ax.imshow(sasImg, cmap=ListedColormap(sasColormap()), aspect='equal', extent=extent, origin='lower')
                plt.title(f't4e2_06 {threshold}th Percentile Mask and Detected O Center')
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Normalized dB')
                ax.contour(np.linspace(0, 50, width), np.linspace(0, 50, height), mask, colors='r', linewidths=1)
                # Convert pixel coordinates to cm for plotting
                x_cm = cx * (50 / width)
                y_cm = cy * (50 / height)
                ax.plot(x_cm, y_cm, 'rx', label='Detected Center')
                # Draw the assumed O as a circle (radius in cm)
                circle = plt.Circle((x_cm, y_cm), radius_cm, color='cyan', fill=False, linewidth=2, label='Assumed O')
                ax.add_patch(circle)
                plt.xlabel('Along-track (cm)', fontsize=10)
                plt.ylabel('Cross-track (cm)', fontsize=10)
                plt.legend()
                plt.show()
            center = (cx, cy)
            return mask, center
    return None, None
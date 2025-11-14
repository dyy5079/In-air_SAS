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

def cropTarget(A, dynamicRange=0, normFlag=True, plot=False, filename=None, output_dir=None, channel=None):
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
    xVec = A.Results.Bp.xVect
    yVec = A.Results.Bp.yVect
    image = A.Results.Bp.image

    # Decode filename if provided
    file = None
    if filename:
        file = fileAttribute(filename)
        # if file:
        #     print(f"File: {filename}")
        #     print(f"  Target Type: {file['target']}")
        #     print(f"  Environment: {file['env']}")
        #     print(f"  Trial: {file['trial']}")

    # Defined chip size from the cfarDetector.m
    chipLx = 0.75
    chipLy = 0.75

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

    # Store all chips (both dB-scaled and complex)
    
    jchips = []
    
    # Process each target
    for i, pos in enumerate(targetPos):
        # Find the indices corresponding to the target position
        xIdx = np.argmin(np.abs(xVec - pos[0]))
        yIdx = np.argmin(np.abs(yVec - pos[1]))
        
        # Crop the complex-valued chip from the original image
        jchip = image[yIdx - halfChipLy:yIdx + halfChipLy, 
                             xIdx - halfChipLx:xIdx + halfChipLx]
        jchips.append(jchip)

    # Save each chip as a separate h5 file if output_dir is provided
    if output_dir and filename and channel is not None:
        for i, (jchip, pos) in enumerate(zip(jchips, targetPos)):
            # Create filename with target number
            target_num = i + 1
            complex_output_path = os.path.join(output_dir, f'{filename[:-3]}_ch{channel}_chip{target_num}.h5')
            
            with h5py.File(complex_output_path, 'w') as hf:
                # Store complex image chip for this target
                hf.create_dataset('chip', data=np.array(jchip), dtype=np.complex64)
                
                # Store target number
                hf.create_dataset('targetNum', data=target_num, dtype=np.int32)
                
                # Store metadata for MATLAB compatibility
                # Store as byte strings for MATLAB compatibility
                hf.create_dataset('target', data=np.bytes_(file['target']))
                hf.create_dataset('env', data=np.bytes_(file['env']))
                hf.create_dataset('trial', data=file['trial'], dtype=np.int32)
                
                # Store this target's position
                hf.create_dataset('targetPos', data=pos, dtype=np.float64)
                
                # Store chip dimensions
                hf.create_dataset('chipX', data=chipLx, dtype=np.float64)
                hf.create_dataset('chipY', data=chipLy, dtype=np.float64)

    # Plot the chips if requested
    if plot:
        # Create individual square plots for each chip 
        if normFlag:
            rNorm = 20 * np.log10(np.tile(yVec, (len(xVec), 1)).T)
        else:
            rNorm = 0
        sasImg = 20 * np.log10(np.abs(image) + 1e-12) + rNorm #adding 1e-12 to avoid log of zero

        dBchips = []
        for i, pos in enumerate(targetPos):
            # Find the indices corresponding to the target position
            xIdx = np.argmin(np.abs(xVec - pos[0]))
            yIdx = np.argmin(np.abs(yVec - pos[1]))

            # Crop the chip from the SAS image (dB-scaled for plotting)
            chip = sasImg[yIdx - halfChipLy:yIdx + halfChipLy, 
                      xIdx - halfChipLx:xIdx + halfChipLx]
            dBchips.append(chip)

        for i, (chip, pos) in enumerate(zip(dBchips, targetPos)):
            # Create a new figure for each chip with square aspect ratio
            fig = plt.figure(i + 1)
            fig.clf()            
            
            # Create extent for proper axis labels
            plt.imshow(chip, 
                          aspect='equal',  # Square aspect ratio
                          cmap=ListedColormap(sasColormap()),
                          extent = [-chipLx/2, chipLx/2, -chipLy/2, chipLy/2],
                          origin='lower')
            
            # Create title with filename info if available
            title = f'Target {i+1} at ({pos[0]:.2f}, {pos[1]:.2f}) m'
            if file:
                title += f"\n{file['target']} - {file['env']} (Trial {file['trial']})"
            plt.title(title, fontsize=10)
            
            plt.xlabel('Along-track (m)', fontsize=10)
            plt.ylabel('Cross-track (m)', fontsize=10)
            clim = [max(chip.max() - dynamicRange, chip.min()), chip.max()]
            plt.clim(clim)
            h = plt.colorbar()
            h.set_label('Amplitude (dB re: 1V @ 1m)')
            
            if output_dir and filename and channel is not None:
                plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_ch{channel}_{i+1}.png'), dpi=300, bbox_inches='tight')
            elif output_dir and filename:
                plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_chip_{i+1}.png'), dpi=300, bbox_inches='tight')

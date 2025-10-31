import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import h5py
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

def kSpaceCrop(A, dynamicRange=0, normFlag=True, plot=False, filename=None, output_dir=None, channel=None):
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

    # Initialize list to hold k-space chips

    kchips = []

    # Compute full k-space (FFT of the entire complex image) and center zero frequency
    kImg = np.fft.fftshift(np.fft.fft2(image))
    
    # Process each target
    for i, pos in enumerate(targetPos):
        # Find the indices corresponding to the target position
        xIdx = np.argmin(np.abs(xVec - pos[0]))
        yIdx = np.argmin(np.abs(yVec - pos[1]))
        
        # Crop the complex-valued chip from the original image
        jchip = image[yIdx - halfChipLy:yIdx + halfChipLy, 
                     xIdx - halfChipLx:xIdx + halfChipLx]
        # k-space of the chip with centering the zero frequency
        kchip = np.fft.fftshift(np.fft.fft2(jchip))
        kchips.append(kchip)

    # Save each k-space chip as a separate h5 file if output_dir is provided
    if output_dir and filename and channel is not None:
        # Save full k-space
        kspace_output_path = os.path.join(output_dir, f'{filename[:-3]}_ch{channel}_kspace.h5')
        with h5py.File(kspace_output_path, 'w') as hf:
                # Store k-space image chip for this target
                hf.create_dataset('kImg', data=np.array(kImg), dtype=np.complex64)
                
                # Store metadata for MATLAB compatibility
                # Store as byte strings for MATLAB compatibility
                hf.create_dataset('target', data=np.bytes_(file['target']))
                hf.create_dataset('env', data=np.bytes_(file['env']))
                hf.create_dataset('trial', data=file['trial'], dtype=np.int32)
                

        for i, (kchip, pos) in enumerate(zip(kchips, targetPos)):
            # Create filename with target number
            target_num = i + 1
            kspace_output_path = os.path.join(output_dir, f'{filename[:-3]}_ch{channel}_kspace_chip{target_num}.h5')
            
            with h5py.File(kspace_output_path, 'w') as hf:
                # Store k-space image chip for this target
                hf.create_dataset('kchips', data=kchip, dtype=np.complex64)
                
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
        kImgdB = 20 * np.log10(np.abs(kImg) + 1e-12)

        chips = []
        for i, pos in enumerate(targetPos):
            # Find the indices corresponding to the target position
            xIdx = np.argmin(np.abs(xVec - pos[0]))
            yIdx = np.argmin(np.abs(yVec - pos[1]))

            # Crop the chip from the SAS image (dB-scaled for plotting)
            chip = sasImg[yIdx - halfChipLy:yIdx + halfChipLy, 
                      xIdx - halfChipLx:xIdx + halfChipLx]
            chips.append(chip)

        # Plot k-space
        fig_kspace_full = plt.figure(len(targetPos) + 1)
        fig_kspace_full.clf()
        plt.imshow(kImgdB,
                   aspect='equal',
                   cmap='viridis',
                   origin='lower')
        title_full = 'Full K-Space (FFT of Complex Image)'
        if file:
            title_full += f"\n{file['target']} - {file['env']} (Trial {file['trial']})"
        plt.title(title_full, fontsize=10)
        plt.xlabel('Frequency X (cycles/pixel)', fontsize=10)
        plt.ylabel('Frequency Y (cycles/pixel)', fontsize=10)
        plt.colorbar(label='Magnitude (dB)')
        if output_dir and filename and channel is not None:
            plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_ch{channel}_kspace.png'), dpi=300, bbox_inches='tight')
        

        # Plot k-space chips
        for i, (chip, pos) in enumerate(zip(chips, targetPos)):

            kchipdB = 20 * np.log10(np.abs(kchips[i]) + 1e-12)
            fig_kspace_chip = plt.figure(len(targetPos) + 2 + i)
            fig_kspace_chip.clf()
            
            # Convert to dB scale for visualization
            
            plt.imshow(kchipdB,
                       aspect='equal',
                       cmap='viridis',
                       origin='lower')
            
            # Create title with filename info if available
            title_kspace = f'K-Space Target {i+1} at ({pos[0]:.2f}, {pos[1]:.2f}) m'
            if file:
                title_kspace += f"\n{file['target']} - {file['env']} (Trial {file['trial']})"
            plt.title(title_kspace, fontsize=10)

            plt.xlabel('Frequency X (cycles/pixel)', fontsize=10)
            plt.ylabel('Frequency Y (cycles/pixel)', fontsize=10)
            plt.colorbar(label='Magnitude (dB)')
            
            if output_dir and filename and channel is not None:
                plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_ch{channel}_kspace_{i+1}.png'), dpi=300, bbox_inches='tight')
            elif output_dir and filename:
                plt.savefig(os.path.join(output_dir, f'{filename[:-3]}_kspace_{i+1}.png'), dpi=300, bbox_inches='tight')

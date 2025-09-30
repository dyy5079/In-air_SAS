import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .sasColormap import sasColormap
#import pdb; pdb.set_trace()
def reconstructImage(A, r, img_plane, fov):
    """
    bpLAirSAS Reconstructs a SAS image using backprojection
        Inputs:
        A = structure with preprocessed data (from pp_airsas)
        r = upsampling ratio
        imgPlane = coordinate along the z-axis at which to beamform
        fov = field of view of transmitter, degrees

        Outputs:
        A = structure with backprojection data added
    """
    c = A.Params.soundSpeed # sound speed, m/s (array-like)
    fs = A.Params.fs         # sampling rate, Hz
    
    # A.Params.position has shape (1, 1001) - we want the number of columns (1001)
    n_pings = A.Params.position.shape[1]  # number of pings is the number of columns
    
    # Resample the data - upsample by factor r
    original_samples = A.Data.tsRC.shape[0]
    new_samples = int(original_samples * r) # new number of samples after resampling
    
    data = resample(A.Data.tsRC, new_samples, axis=0)
    recording_scope = data.shape[0]

    # Define the scene extent
    x = A.Results.Bp.xVect
    y = A.Results.Bp.yVect
    img = np.zeros((len(y), len(x)), dtype=np.complex128)

    # All pixel coordinates
    Xs, Ys = np.meshgrid(x, y)

    # Backprojection (delay and sum)
    #for ping in range(n_pings):
    for ping in range(1):
        # A.Params.position has shape (1, 1001), so we access [0, ping] to get the ping-th position
        # Progress tracking to detect infinite loops
        if ping % 100 == 0:  # Print every 100 iterations
            print(f"Processing ping {ping}/{n_pings} ({ping/n_pings*100:.1f}%)")
    
        m_position_value = A.Params.position[0, ping]
        
        position_offset = np.array([m_position_value, 0, 0])  # x-axis movement only

        Rx = A.Hardware.rxPos + position_offset
        Tx = A.Hardware.txPos + position_offset

        # Compute distance from Tx to each pixel and back to Rx
        distMtx = (
            np.sqrt((Xs - Tx[0])**2 + (Ys - Tx[1])**2 + (img_plane - Tx[2])**2) + np.sqrt((Xs - Rx[0])**2 + (Ys - Rx[1])**2 + (img_plane - Rx[2])**2)
        )

        thetaMtx = np.degrees(np.arctan((Xs - Tx[0]) / (Ys - Tx[1])))

        # Nearest neighbor interpolation
        time_indices = np.round(distMtx / c[ping] * fs * r).astype(int)
        times = np.minimum(time_indices, recording_scope-1)

        # Isolate the ping
        p_data = data[:, ping]
        print(p_data.shape)
        ping_contributions = p_data[times]

        # Mask out-of-scope and out-of-FOV values
        ping_contributions = ping_contributions * (time_indices < recording_scope) * (np.abs(thetaMtx) < fov / 2)

        # Accumulate energy
        img = img + ping_contributions

    # Pass the image to the data structure
    A.Results.Bp.image = img
    return A
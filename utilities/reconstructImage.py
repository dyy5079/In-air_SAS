import numpy as np
from scipy.signal import resample

def reconstruct_image(A, r, img_plane, fov):
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
    
    n_pings = len(A.Params.position)  # number of pings

    # Resample the data
    data = resample(A.Data.tsRC, r, 1, axis=0)
    recording_scope = data.shape[0]

    # Define the scene extent
    x = A.Results.Bp.xVect
    y = A.Results.Bp.yVect
    img = np.zeros((len(y), len(x)), dtype=np.complex128)

    # All pixel coordinates
    Xs, Ys = np.meshgrid(x, y)

    # Backprojection (delay and sum)
    for ping in range(n_pings):
        m_position = A.Params.position[ping]
        Rx = A.Hardware.rxPos + np.array([m_position, 0, 0])
        Tx = A.Hardware.txPos + np.array([m_position, 0, 0])

        # Compute distance from Tx to each pixel and back to Rx
        distMtx = (
            np.sqrt((Xs - Tx[0])**2 + (Ys - Tx[1])**2 + (img_plane - Tx[2])**2) + np.sqrt((Xs - Rx[0])**2 + (Ys - Rx[1])**2 + (img_plane - Rx[2])**2)
        )

        thetaMtx = np.degrees(np.arctan((Xs - Tx[0]) / (Ys - Tx[1])))

        # Nearest neighbor interpolation
        time_indices = np.round(distMtx / c[ping] * fs * r).astype(int)
        times = np.minimum(time_indices, recording_scope)

        # Isolate the ping
        p_data = data[:, ping]
        ping_contributions = p_data[times]

        # Mask out-of-scope and out-of-FOV values
        ping_contributions = ping_contributions * (time_indices < recording_scope) * (np.abs(thetaMtx) < fov / 2)

        # Accumulate energy
        img += ping_contributions

    # Pass the image to the data structure
    A.Results.Bp.image = img
    return A
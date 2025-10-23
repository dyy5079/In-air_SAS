import numpy as np
import h5py
import pandas as pd
import os
from scipy.signal import hilbert, lfilter, remez
from scipy.signal.windows import tukey
from scipy.signal import kaiserord

from .initStruct import initStruct, singleEmptyStruct
from .getAirSpeed import getAirSpeed
from .genLfm import genLfm
from .freqVecGen import freqVecGen

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .sasColormap import sasColormap

class P:
    def __init__(self):
        self.temp = np.nan
        self.humidity = np.nan
        self.position = np.nan
        self.soundSpeed = np.nan
        self.time = np.nan
        self.fs = np.nan
        self.nSamples = np.nan

def AirsasHpf(filterStop, filterPass, fs):
    Dstop = 0.01
    Dpass = 0.005756
    # Dpass = 0.0057563991496
    # normalizedFilterStop = filterStop / (fs / 2.0)
    # normalizedFilterPass = filterPass / (fs / 2.0)

    #bands = [0, (filterStop/(fs/2.0)), (filterPass/(fs/2.0)), 2.0]
    # weights = [Dstop, Dpass]

    # b = remez(numtaps, bands, desired, weight=weights, fs=2.0)
    
    # Convert to remez parameters
    nyq = fs / 2.0  # new line
    bands = [0, filterStop, filterPass, nyq]  # new line
    desired = [0, 1]  # new line
    weights = [1/Dstop, 1/Dpass]  # new line
    N = 107
    b = remez(N, bands, desired, weight=weights, fs=fs)  # new line
    
    
    return b

# This subfunction replica correlates the data with the transmitted
# waveform to perform pulse compression.  The resulting data is output as a
# complex-valued timeseries.

def mfilt(data, pulseReplica):
    
    #data = hilbert_matlab(np.real(data))    # Use the custom hilbert function to replicate MATLAB's behavior
    data = hilbert(data, axis=0)  # Use scipy's hilbert function along axis 0

    nPtsData = data.shape[0]
    nPtsMfk = len(pulseReplica)

    data[:nPtsMfk] = 0

    nFFT = nPtsData + nPtsMfk - 1
    filterKernelF = np.fft.fft(hilbert(pulseReplica), nFFT)

    dataF = np.fft.fft(data, nFFT, axis=0)
    # Reshape filterKernelF to be compatible with 2D dataF
    filterKernelF = filterKernelF.reshape(-1, 1)
    dataF = dataF * np.conj(filterKernelF)
    data = np.fft.ifft(dataF, axis=0)
    data = data[:nPtsData, :]
    return data

# This sub-function removes the the group delay of the data acquisition
# system. It ensures that real input data results in real output data.
def removeGroupDelay(A):
    data = np.fft.fft(A.Data.tsRC, axis=0)
    nT = data.shape[0]

    freqVec = freqVecGen(nT, A.Params.fs).T
    groupDelay_s = A.Hardware.groupDelay / A.Params.fs

    delayRamp = np.exp(1j * 2 * np.pi * freqVec * groupDelay_s)
    
    if nT % 2 == 0:
        delayRamp[(nT // 2)] = np.real(delayRamp[(nT // 2)])
    delayRamp = delayRamp.reshape(-1, 1)                        # Reshape delayRamp to be a column vector
    A.Data.tsRC = np.real(np.fft.ifft(data * delayRamp, axis=0))

    return A

# This subfunction zeros out ("blanks") the portion of the received waveforms that
# correspond to when the transmitter was emitting the pulse.
def txBlanker(A):
    
    pulseReplicaLength = int(A.Wfm.pulseLength * A.Params.fs)
    
    if pulseReplicaLength % 1 != 0:
        raise ValueError('PP_AIRSAS: Pulse length not an integer sample length')
    
    blankLength = pulseReplicaLength

    blanker = np.ones((A.Data.tsRC.shape[0], 1))
    blanker[:blankLength, 0] = 0  # Set first blankLength rows to 0
    blanker[blankLength:blankLength + pulseReplicaLength, 0] = (np.sin(np.linspace(-np.pi/2, np.pi/2, pulseReplicaLength)) + 1) / 2
    A.Data.tsRC = np.multiply(A.Data.tsRC, blanker)
    return A

def packToStruct(folder, filename, chanSelect, cSelect):
    # Pre-populate the data structure
    A = initStruct(len(chanSelect))

    dPath = os.path.join(folder, "scenes", filename)

    # Load non-acoustic parameters
    with h5py.File(dPath, 'r') as f:
        params = P()
        params.temp = np.array(f['/na/temperature'])
        params.humidity = np.array(f['/na/humidity'])
        params.position = np.array(f['/na/position'])
    params.soundSpeed = getAirSpeed(params, cSelect)
 
    # Acquisition parameters
    if isfile := os.path.isfile(os.path.join(folder, "characterization data", "acquisitionParams.csv")):
        print(f"Loading acquisition parameters from {os.path.join(folder, 'characterization data', 'acquisitionParams.csv')}")
        daqParams = pd.read_csv(os.path.join(folder, "characterization data", "acquisitionParams.csv"))
    elif isfile := os.path.isfile(os.path.join(folder, "characterization data", "acquistionParams.csv")):
        print(f"Loading acquisition parameters from {os.path.join(folder, 'characterization data', 'acquistionParams.csv')}")
        daqParams = pd.read_csv(os.path.join(folder, "characterization data", "acquistionParams.csv"))
    else:
        print(f"Error: acquisitionParams.csv not found in {os.path.join(folder, 'characterization data')}")
        exit()

    params.fs = float(daqParams.iloc[0, 1])
    # For collection_date, you may need to use h5py's attrs
    with h5py.File(dPath, 'r') as f:
        params.time = f.attrs['collection_date']
    for i in range(len(A)):
        A[i].Params = params
    groupDelay = float(daqParams.iloc[1, 1])

    if isfile := os.path.isfile(os.path.join(folder, "characterization data", "sensorCoordinates.csv")):
        print(f"Loading sensor coordinates from {os.path.join(folder, 'characterization data', 'sensorCoordinates.csv')}")
        sensorCoord = pd.read_csv(os.path.join(folder, "characterization data", "sensorCoordinates.csv"), index_col=0).to_numpy()
    else:
        print(f"Error: sensorCoordinates.csv not found in {os.path.join(folder, 'characterization data')}")
        exit()
    for n in range(len(chanSelect)):
        A[n].Hardware.rxPos = sensorCoord[chanSelect[n], :]  # chanSelect[n] directly indexes the receiver
        A[n].Hardware.txPos = sensorCoord[0, :]  # Row 0 is the transmitter
        A[n].Hardware.groupDelay = float(groupDelay)

    # Waveform parameters
    temp_struct = singleEmptyStruct()
    wfm = temp_struct.Wfm.__class__()  # Create a new instance of the Wfm class
    wfm.pulseType = 'LFM'
    wfm.fStart = float(daqParams.iloc[7, 1])
    wfm.fStop = float(daqParams.iloc[8, 1])
    wfm.pulseLength = float(daqParams.iloc[9, 1])
    wfm.amplitude = float(daqParams.iloc[10, 1])
    pulseReplica = genLfm(float(wfm.fStart), float(wfm.fStop) - float(wfm.fStart), float(wfm.pulseLength), float(params.fs))
    win = tukey(int(float(wfm.pulseLength) * float(params.fs)), float(daqParams.iloc[12, 1]))
    wfm.pulseReplica = pulseReplica * win
    for i in range(len(A)):
        A[i].Wfm = wfm  # Remove .copy() as it may not be available


    
    # Pack the time series data
    for n in range(len(chanSelect)):
    #for n in range(1):
        with h5py.File(dPath, 'r') as f:
            A[n].Data.tsRaw = np.array(f[f"/ch{chanSelect[n]}/ts"])
        A[n].Data.tsRC = A[n].Data.tsRaw.copy()
        A[n].Data.tsRC = A[n].Data.tsRC.T
        
        if n == 1:
            plt.figure()
            plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12).T, aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()), vmin=-60, vmax=0)
            h = plt.colorbar()
            h.set_label('Amplitude dB')
            plt.xlabel('Samples')
            plt.ylabel('Pings')
            plt.title('Original(tsRaw)')
            plt.gca().invert_yaxis()
            plt.show()
        # make sure to remove the mean from each ping separately
        # remove DC bias from each individual ping rather than a global DC offset.
        A[n].Data.tsRC = A[n].Data.tsRC - np.mean(A[n].Data.tsRC, axis=0, keepdims=True)
        
        if n == 1:
            plt.figure()
            plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12).T, aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()), vmin=-60, vmax=0)
            h = plt.colorbar()
            h.set_label('Amplitude dB')
            plt.xlabel('Samples')
            plt.ylabel('Pings')
            plt.title('After Mean Removal')
            plt.gca().invert_yaxis()
            plt.show()
        # Remove the group delay of the acquisition system
        A[n] = removeGroupDelay(A[n])
        if n == 1:
            plt.figure()
            plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12).T, aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()), vmin=-60, vmax=0)
            h = plt.colorbar()
            h.set_label('Amplitude dB')
            plt.xlabel('Samples')
            plt.ylabel('Pings')
            plt.title('After Group Delay Removal')
            plt.gca().invert_yaxis()
            plt.show()
        # Remove the direct path transmission from speaker to microphone
        A[n] = txBlanker(A[n])

        if n == 1:
            plt.figure()
            plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12).T, aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()), vmin=-60, vmax=0)
            h = plt.colorbar()
            h.set_label('Amplitude dB')
            plt.xlabel('Samples')
            plt.ylabel('Pings')
            plt.title('After Blanking')
            plt.gca().invert_yaxis()
            plt.show()
        # Apply a bandpass filter
        bandEdge = min([A[n].Wfm.fStart, A[n].Wfm.fStop])  # Changed from dictionary access to attribute access
        if bandEdge >= 5e3:
            b = AirsasHpf(bandEdge-2e3, bandEdge, A[n].Params.fs)
            A[n].Data.tsRC = lfilter(b, 1, A[n].Data.tsRC, axis=0)
            A[n].Data.tsRC = np.roll(A[n].Data.tsRC, -int((len(b)-1)/2), axis=0)
        
        if n == 1:
            plt.figure()
            plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12).T, aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()), vmin=-60, vmax=0)
            h = plt.colorbar()
            h.set_label('Amplitude dB')
            plt.xlabel('Samples')
            plt.ylabel('Pings')
            plt.title('After Bandpass Filter')
            plt.gca().invert_yaxis()
            plt.show()
        A[n].Data.tsRC = mfilt(A[n].Data.tsRC, A[n].Wfm.pulseReplica)

        if n == 1:
            plt.figure()
            plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12).T, aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()), vmin=-60, vmax=0)
            h = plt.colorbar()
            h.set_label('Amplitude dB')
            plt.xlabel('Samples')
            plt.ylabel('Pings')
            plt.title('After Matched Filter')
            plt.gca().invert_yaxis()
            plt.show()
        
        # plt.tight_layout()
        # plt.savefig("BeforeAfterPlot", dpi=300, bbox_inches='tight')
        # plt.show()
    return A


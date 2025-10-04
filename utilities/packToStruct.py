import numpy as np
import h5py
import pandas as pd
import os
from scipy.signal import hilbert, lfilter, remez
from scipy.signal.windows import tukey

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

    nyq = fs / 2.0
    bands = [0, filterStop, filterPass, nyq]
    desired = [0, 1]
    weights = [Dstop, Dpass]

    numtaps = 101

    b = remez(numtaps, bands, desired, weight=weights, fs=fs)
    return b


def mfilt(data, pulseReplica):
    # Only apply Hilbert transform if data is real
    if np.isrealobj(data):
        data = hilbert(data)
    # If data is already complex, use it as-is

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

def removeGroupDelay(A):
    data = np.fft.fft(A.Data.tsRC, axis=0)
    nT = data.shape[0]

    freqVec = freqVecGen(nT, A.Params.fs)
    groupDelay_s = A.Hardware.groupDelay / A.Params.fs

    delayRamp = np.exp(1j * 2 * np.pi * freqVec * groupDelay_s)
    if nT % 2 == 0:
        delayRamp[nT // 2] = np.real(delayRamp[nT // 2])
    
    # Reshape delayRamp to be compatible with 2D data
    delayRamp = delayRamp.reshape(-1, 1)

    A.Data.tsRC = np.fft.ifft(data * delayRamp, axis=0)
    return A

def txBlanker(A):
    pulseReplicaLength = int(A.Wfm.pulseLength * A.Params.fs)
    if pulseReplicaLength % 1 != 0:
        raise ValueError('PP_AIRSAS: Pulse length not an integer sample length')
    blankLength = pulseReplicaLength
    blanker = np.ones(A.Data.tsRC.shape)
    blanker[:blankLength] = 0
    blanker[blankLength:blankLength + pulseReplicaLength, :] = (
        (np.sin(np.linspace(-np.pi/2, np.pi/2, pulseReplicaLength)) + 1) / 2
    ).reshape(-1, 1)
    A.Data.tsRC = A.Data.tsRC * blanker
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
    #for n in range(len(chanSelect)):
    for n in range(1):
        with h5py.File(dPath, 'r') as f:
            A[n].Data.tsRaw = np.array(f[f"/ch{chanSelect[n]}/ts"])
        A[n].Data.tsRC = A[n].Data.tsRaw.copy()
        plt.subplot(2, 2, 1)
        plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12), aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()))
        h = plt.colorbar()
        plt.title('Original')
        plt.gca().invert_yaxis()

        A[n].Data.tsRC -= np.mean(A[n].Data.tsRC)
        
        plt.subplot(2, 2, 2)
        plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12), aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()))
        h = plt.colorbar()
        plt.title('After Mean Removal')
        plt.gca().invert_yaxis()

        A[n] = removeGroupDelay(A[n])
        A[n] = txBlanker(A[n])
        plt.subplot(2, 2, 3)
        plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12), aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()))
        plt.title('After Blanking and removedelay')
        h = plt.colorbar()
        plt.gca().invert_yaxis()
        
        bandEdge = min([A[n].Wfm.fStart, A[n].Wfm.fStop])  # Changed from dictionary access to attribute access
        if bandEdge >= 5e3:
            b = AirsasHpf(bandEdge-2e3, bandEdge, A[n].Params.fs)
            A[n].Data.tsRC = lfilter(b, 1, A[n].Data.tsRC)
            A[n].Data.tsRC = np.roll(A[n].Data.tsRC, -int((len(b)-1)/2), axis=0)
        A[n].Data.tsRC = mfilt(A[n].Data.tsRC, A[n].Wfm.pulseReplica)

        plt.subplot(2, 2, 4)
        plt.imshow(20 * np.log10(np.abs(A[n].Data.tsRC) + 1e-12), aspect='auto', origin='lower', cmap=ListedColormap(sasColormap()))
        plt.title('After Matched Filter')
        h = plt.colorbar()
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    return A


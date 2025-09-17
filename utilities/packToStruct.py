import numpy as np
import h5py
import pandas as pd
from scipy.signal import tukey, hilbert, firwin, lfilter
from initStruct import initStruct, Wfm
from getAirSpeed import getAirSpeed
from genLfm import genLfm
from freqVecGen import freqVecGen

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
    # Hilbert transform for analytic signal
    data = hilbert(data)

    nPtsData = data.shape[0]
    nPtsMfk = len(pulseReplica)

    data[:nPtsMfk] = 0

    nFFT = nPtsData + nPtsMfk - 1
    filterKernelF = np.fft.fft(hilbert(pulseReplica), nFFT)

    dataF = np.fft.fft(data, nFFT, axis=0)
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
        delayRamp[nT // 2 + 1] = np.real(delayRamp[nT // 2])

    A.Data.tsRC = np.fft.ifft(data * delayRamp, axis=0)
    return A

def txBlanker(A):
    pulseReplicaLength = int(A.Wfm['pulseLength'] * A.Params.fs)
    if pulseReplicaLength % 1 != 0:
        raise ValueError('PP_AIRSAS: Pulse length not an integer sample length')
    blankLength = pulseReplicaLength
    blanker = np.ones(A.Data.tsRC.shape)
    blanker[:blankLength] = 0
    blanker[blankLength:blankLength + pulseReplicaLength] = (
        (np.sin(np.linspace(-np.pi/2, np.pi/2, pulseReplicaLength)) + 1) / 2
    )
    A.Data.tsRC = A.Data.tsRC * blanker
    return A



def packToStruct(folder, filename, chanSelect, cSelect):
    # Pre-populate the data structure
    A = initStruct(len(chanSelect))

    dPath = f"{folder}/scenes/{filename}"

    # Load non-acoustic parameters
    with h5py.File(dPath, 'r') as f:
        P = P()
        P.temp = np.array(f['/na/temperature'])
        P.humidity = np.array(f['/na/humidity'])
        P.position = np.array(f['/na/position'])
    P.soundSpeed = getAirSpeed(P, cSelect)

    # Acquisition parameters
    daqParams = pd.read_csv(f"{folder}/characterization data/acquistionParams.csv", header=None)
    P.fs = daqParams.iloc[0, 1]
    # For collection_date, you may need to use h5py's attrs
    with h5py.File(dPath, 'r') as f:
        P.time = f.attrs['collection_date']
    for i in range(len(A)):
        A[i].Params = P
    groupDelay = daqParams.iloc[1, 1]

    sensorCoord = np.loadtxt(f"{folder}/characterization data/sensorCoordinates.csv", delimiter=',')
    for n in range(len(chanSelect)):
        A[n].Hardware.rxPos = sensorCoord[chanSelect[n]+1, 1:4]
        A[n].Hardware.txPos = sensorCoord[0, 1:4]
        A[n].Hardware.groupDelay = groupDelay

    # Waveform parameters
    Wfm = Wfm()
    Wfm.pulseType = 'LFM'
    Wfm.fStart = daqParams.iloc[7, 1]
    Wfm.fStop = daqParams.iloc[8, 1]
    Wfm.pulseLength = daqParams.iloc[9, 1]
    Wfm.amplitude = daqParams.iloc[10, 1]
    pulseReplica = genLfm(Wfm.fStart, Wfm.fStop - Wfm.fStart, Wfm.pulseLength, P.fs)
    win = tukey(int(Wfm.pulseLength * P.fs), daqParams.iloc[12, 1])
    Wfm.pulseReplica = pulseReplica * win
    for i in range(len(A)):
        A[i].Wfm = Wfm.copy()

    # Pack the time series data
    for n in range(1, len(A)):
        with h5py.File(dPath, 'r') as f:
            A[n].Data.tsRaw = np.array(f[f"/ch{chanSelect[n]}/ts"])
        A[n].Data.tsRC = A[n].Data.tsRaw.copy()
        A[n].Data.tsRC -= np.mean(A[n].Data.tsRC)
        A[n] = removeGroupDelay(A[n])
        A[n] = txBlanker(A[n])
        bandEdge = min([A[n].Wfm['fStart'], A[n].Wfm['fStop']])
        if bandEdge >= 5e3:
            b = AirsasHpf(bandEdge-2e3, bandEdge, A[n].Params.fs)
            A[n].Data.tsRC = lfilter(b, 1, A[n].Data.tsRC)
            A[n].Data.tsRC = np.roll(A[n].Data.tsRC, -int((len(b)-1)/2), axis=0)
        A[n].Data.tsRC = mfilt(A[n].Data.tsRC, A[n].Wfm.pulseReplica)
    return A


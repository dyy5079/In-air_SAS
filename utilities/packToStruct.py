import numpy as np
import h5py
import pandas as pd
from scipy.signal import tukey, hilbert, firwin, lfilter
from initStruct import initStruct
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


def pack_to_struct(folder, filename, chanSelect, cSelect):
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
    for a in A:
        a.Params = P
    groupDelay = daqParams.iloc[1, 1]

    sensorCoord = np.loadtxt(f"{folder}/characterization data/sensorCoordinates.csv", delimiter=',')
    for n in range(len(chanSelect)):
        a
        a.Hardware.rxPos = sensorCoord[chanSelect[n]+1, 1:4]
        a.Hardware.txPos = sensorCoord[0, 1:4]
        a.Hardware.groupDelay = groupDelay

    # Waveform parameters
    Wfm = {}
    Wfm['pulseType'] = 'LFM'
    Wfm['fStart'] = daqParams.iloc[7, 1]
    Wfm['fStop'] = daqParams.iloc[8, 1]
    Wfm['pulseLength'] = daqParams.iloc[9, 1]
    Wfm['amplitude'] = daqParams.iloc[10, 1]
    pulseReplica = genLfm(Wfm['fStart'], Wfm['fStop'] - Wfm['fStart'], Wfm['pulseLength'], P.fs)
    win = tukey(int(Wfm['pulseLength'] * P.fs), daqParams.iloc[12, 1])
    Wfm['pulseReplica'] = pulseReplica * win
    for a in A:
        a.Wfm = Wfm.copy()

    # Pack the time series data
    for n, a in enumerate(A):
        with h5py.File(dPath, 'r') as f:
            a.Data.tsRaw = np.array(f[f"/ch{chanSelect[n]}/ts"])
        a.Data.tsRC = a.Data.tsRaw.copy()
        a.Data.tsRC -= np.mean(a.Data.tsRC)
        a = removeGroupDelay(a)
        a = txBlanker(a)
        bandEdge = min([a.Wfm['fStart'], a.Wfm['fStop']])
        if bandEdge >= 5e3:
            # Use firwin as a placeholder for AirsasHpf
            nyq = a.Params.fs / 2
            b = firwin(101, (bandEdge - 2e3) / nyq, pass_zero=False)
            a.Data.tsRC = lfilter(b, 1, a.Data.tsRC)
            a.Data.tsRC = np.roll(a.Data.tsRC, -int((len(b)-1)/2))
        a.Data.tsRC = mFilt(a.Data.tsRC, a.Wfm['pulseReplica'])
    return A
    return A

def mFilt(data, pulseReplica):
    # Hilbert transform for analytic signal
    data = hilbert(data)
    nPtsData = data.shape[0]
    nPtsMfk = len(pulseReplica)
    data[:nPtsMfk] = 0
    nFft = nPtsData + nPtsMfk - 1
    filterKernelF = np.fft.fft(hilbert(pulseReplica), nFft)
    dataF = np.fft.fft(data, nFft)
    dataF = dataF * np.conj(filterKernelF)
    data = np.fft.ifft(dataF)[:nPtsData]
    return data

def removeGroupDelay(a):
    data = np.fft.fft(a.Data.tsRC)
    nT = data.shape[0]
    freqVec = freqVecGen(nT, a.Params.fs)
    groupDelayS = a.Hardware.groupDelay / a.Params.fs
    delayRamp = np.exp(1j * 2 * np.pi * freqVec * groupDelayS)
    if nT % 2 == 0:
        delayRamp[nT // 2] = np.real(delayRamp[nT // 2])
    a.Data.tsRC = np.fft.ifft(data * delayRamp)
    return a

def txBlanker(a):
    pulseReplicaLength = int(a.Wfm['pulseLength'] * a.Params.fs)
    if pulseReplicaLength % 1 != 0:
        raise ValueError('PP_AIRSAS: Pulse length not an integer sample length')
    blankLength = pulseReplicaLength
    blanker = np.ones(a.Data.tsRC.shape)
    blanker[:blankLength] = 0
    blanker[blankLength:blankLength + pulseReplicaLength] = (
        (np.sin(np.linspace(-np.pi/2, np.pi/2, pulseReplicaLength)) + 1) / 2
    )
    a.Data.tsRC = a.Data.tsRC * blanker
    return a

# You will need to implement or adapt these:
# - init_struct
# - get_air_speed
# - gen_lfm
# - freq_vec_gen

# For AirsasHpf, a full FIRPM equivalent is not available in scipy, so firwin is used as a placeholder.
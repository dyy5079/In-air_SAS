import numpy as np

def singleEmptyStruct():
    # Output Image with Backprojection (BP)
    class Bp:
        def __init__(self):
            self.xVect = np.nan             # x coordinates of image pixels
            self.yVect = np.nan             # y coordinates of image pixels
            self.image = np.nan             # output image

    # Sensor Data
    class Data:
        def __init__(self):
            self.tsRaw = np.nan             # Raw recorded time series [nSamples x nPings]
            self.tsRC = np.nan              # Data conditioned and replica correlated

    # Acquisitions & Environment Parameters
    class Params:
        def __init__(self):
            self.fs = np.nan                # Sample rate [Hz]
            self.time = np.nan              # Time [sec]
            self.position = np.nan          # Position on track [m]
            self.nSamples = np.nan          # Number of time sample recorded
            self.nBits = np.nan             # Bit depth of ADC
            self.nAverages = np.nan         # Number of averages taken at each position
            self.temp = np.nan              # nPosition x 2 vector of air tempearture at each measurement [C]
            self.humidity = np.nan          # nPosition x 1 vector of relative humidity at each measurement [%]
            self.soundSpeed = np.nan        # Calculated sound speed in chamber [m/s]
            self.sysUnits = 'mks'           # System of units for variables
            self.angleUnits = 'deg'         # Units for angle variables

    # Waveform Parameters
    class Wfm:
        def __init__(self):
            self.pulseReplica = np.nan      # Time series of the transmitted waveform
            self.fStart = np.nan            # Start frequency [Hz]
            self.fStop = np.nan             # Stop frequenct [Hz]
            self.pulseLength = np.nan       # Length of transmitted waveform [s]
            self.amplitude = np.nan         # Amplitude of the transmitted waveform at the output of the DAQ [V]
            self.winType = np.nan           # String indicating type of window applied to the transmitted waveform. 
                                            # This string should be tha smae as the MATLAB command used to produce 
                                            # the window function
            self.winParams = np.nan         # Parameters passed to the windowing function 
                                            # applied to the transmitted waveform. The parameter order should 
                                            # match that of the MATLAB function used to create the window

    # Hardware Parameters
    class Hardware:
        def __init__(self):
            self.txPos = np.nan             # Transmitter position [m] [x, y, z]
            self.rxPos = np.nan             # Receiver position [m] [x, y, z]
            self.groupDelay = np.nan        # Delay between start of recording and transmission [samples]

    # Output Image with Backprojection (BP)
    class Results:
        def __init__(self):
            self.Bp = Bp()


    class X:
        def __init__(self):
            self.Data = Data()
            self.Params = Params()
            self.Wfm = Wfm()
            self.Hardware = Hardware()
            self.Results = Results()

    return X()

def initStruct(nChans):
    A = []
    for nDexChan in range(nChans):
        A.append(singleEmptyStruct())
    return A
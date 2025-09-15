import numpy as np

def gen_lfm(f_start, bw, tau, fs):
    """
    GENLFM Generates a real-valued linear frequency modulated (LFM) chirp
    
    fStart = starting frequency of chirp, Hz
    bw = bandwidth of chirp (negative number for downchirp), Hz
    tau = duration of chirp, s
    fs = sampling rate, Hz
    sig = real-valued LFM chirp
    """
    n_samp = int(round(tau * fs))  # number of samples
    t = np.arange(n_samp) / fs     # time vector
    B = 0.5 * bw / tau             # frequency modulation factor
    sig = np.sin(2 * np.pi * (f_start + B * t) * t)
    return sig
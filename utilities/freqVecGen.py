import numpy as np

def freqVecGen(Nt, fs):
    """
    FREQVECGEN Generates a vector of frequencies that correspond to the
    elements of an FFT with Nt samples at fs sampling rate.  This function
    works for both odd and even length signals.  The frequencies are arranged
    in order of positive frequencies (0 to fs/2) followed by the negative
    frequencies (-fs/2 to 0-df)

    Nt = Number of samples
    fs - Sampling frequency, Hz
    freqVec - Vector of frequencies
    """
    
    df = fs / Nt  # frequency step

    if Nt % 2 == 0:  # even number of points
        freq_vec = np.arange(-Nt//2, Nt//2) * df
    else:  # odd number of points
        Nt_adj = Nt - 1
        freq_vec = np.arange(-Nt_adj//2, Nt_adj//2 + 1) * df

    pos_freq_mask = freq_vec >= 0
    freq_vec = np.concatenate((freq_vec[pos_freq_mask], freq_vec[~pos_freq_mask]))

    return freq_vec
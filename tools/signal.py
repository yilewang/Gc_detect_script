#!/bin/usr/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import h5py

def hdf5Reader(filename):
    with h5py.File(filename, "r") as f:
    ### get key of h5 file
        key = list(f.keys())
        dset = f[key[0]][:]
        return dset


def psd(signal, samplinginterval, visual=False):
    """
    This function is for power spectrum density analysis
    Input:
        Signal, sampling interveral (1/fs), visual (default is False)
    Output:
        Freq axis, PSD of the signal
    """
    fourierSignal = np.fft.fft(np.array(signal) - np.array(signal).mean())
    spectrum = 2 * (samplinginterval) ** 2 / 1000 * (fourierSignal * fourierSignal.conj())
    spectrum = spectrum[:int(len(np.array(signal)) / 2)] 
    time_all = 1 / 1
    fNQ = 1/samplinginterval/2 # Nyquist frequency
    faxis = np.arange(0, fNQ, time_all) # frequency axis
    if visual:
        fig, axs = plt.subplots()
        psd_freq, psd_signal = psd(np.array(signal), samplinginterval)
        axs.plot(psd_freq, psd_signal, color='r', label = 'PSD Results')
        axs.legend()
        plt.show()
    return faxis, spectrum.real



def fir_bandpass(data, fs, cut_off_low, cut_off_high, width=2.0, ripple_db=10.0):
    """
    The FIR bandpass filter
    Args:
        data: 1-d array
        fs: frequency (sampling rate)
        cut_off_low: the low threshold
        cut_off_high: the high threshold
        width: the time windows for filtering
    Return:
        filtered data, N, delay (for plotting)
        when plot, align the axis by `plt.plot(time[N-1:]-delay, filtered_data[N-1:])`
    """
    nyq_rate = fs / 2.0
    wid = width/nyq_rate
    N, beta = signal.kaiserord(ripple_db, wid)
    taps = signal.firwin(N, cutoff = [cut_off_low, cut_off_high],
                  window = 'hamming', pass_zero = False, fs=fs)
    filtered_signal = signal.lfilter(taps, 1.0, data)
    delay = 0.5 * (N-1) / fs
    return filtered_signal, N, delay


def freqCount(signal, filter=False):
    pCNGRTheta, N, delay= fir_bandpass(dset[:,5], 81920., 2., 10.)
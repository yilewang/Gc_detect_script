#!/bin/usr/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal




def psd(signal, samplinginterval, fs):
    fourierSignal = np.fft.fft(np.array(signal) - np.array(signal).mean())
    spectrum = 2 * (1/fs) ** 2 / 1000 * (fourierSignal * fourierSignal.conj())
    spectrum = spectrum[:int(len(np.array(signal)) / 2)] 
    time_all = 1 / 1
    fNQ = 1/samplinginterval/2 # Nyquist frequency
    faxis = np.arange(0, fNQ, time_all) # frequency axis
    return faxis, spectrum.real

def psd_plot(dfR, dfL, samplinginterval):
    fig, axs = plt.subplots(2, figsize=(15,7))
    psd_freq_right, psd_right = psd(np.array(dfR), samplinginterval)
    psd_freq_left, psd_left = psd(np.array(dfL), samplinginterval)
    axs[0].plot(psd_freq_left, psd_left, color= 'k', label = 'PCG Left')
    axs[0].plot(psd_freq_right, psd_right, color='r', label = 'PCG Right')
    axs[0].set(xlim=[0,100])
    axs[0].legend()


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
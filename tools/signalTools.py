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


def psd(data, samplinginterval, visual=False, xlim=100., *args, **kwargs):
    """
    This function is for power spectrum density analysis
    Input:
        Signal, sampling interveral (1/fs), visual (default is False)
    Output:
        Freq axis, PSD of the signal
    """
    total = len(data)
    duration = total * samplinginterval
    fourierSignal = np.fft.fft(np.array(data) - np.array(data).mean())
    spectrum = 2 * (samplinginterval) ** 2 / duration * (fourierSignal * fourierSignal.conj())
    spectrum = spectrum[:int(len(np.array(data)) / 2)]
    time_all = 1 / duration
    fNQ = 1/samplinginterval/2 # Nyquist frequency
    faxis = np.arange(0, fNQ, time_all) # frequency axis
    if visual:
        fig, axs = plt.subplots()
        axs.plot(faxis, spectrum.real, color='r', label = 'PSD Results', *args, **kwargs)
        axs.legend()
        axs.set_xlim([0, xlim])
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


def freqCount(data, prominence, fs, filter=False, highpass = 2., lowpass = 10.,visual = False, length=10, height=5, dpi=50, *args, **kwargs):
    """
    The *args argument in a function definition allows 
    the function to process an unspecified number of positional arguments.
    
    You can use the *args notation to “unpack” a sequence 
    into a series of positional arguments for any function. 

    **kwargs: To do this, you must package the keyword-value pairs in a dictionary.
    """
    data = np.array(data)
    time = np.arange(0, len(data)/fs, 1/fs)
    spikesdata, _ = signal.find_peaks(data, prominence = prominence)

    if filter:
        postfilter, N, delay= fir_bandpass(data, fs, highpass, lowpass)
        spikesfiltered, _ = signal.find_peaks(postfilter, prominence = prominence)
    # visualization
    if visual:
        fig, ax = plt.subplots(figsize=(length, height), dpi = dpi)
        ax.plot(time, data, label = "signal", *args, **kwargs)
        ax.plot(spikesdata/fs, data[spikesdata], '+', label = "signal spikes", *args, **kwargs)
        if filter:
            ax.plot(time[N-1:]-delay, postfilter[N-1:], label = "filtered signal", *args, **kwargs)
            if len(spikesfiltered) > 0:
                ax.plot(spikesfiltered[spikesfiltered > N-1]/fs - delay, postfilter[spikesfiltered[spikesfiltered > N-1]],'x', label = "filtered spikes", *args, **kwargs)
        ax.legend()
        plt.show()

    # return output
    if filter:
        return (len(spikesdata), len(spikesfiltered))
    else:
        return len(spikesdata)


def ampHilbert(data, fs, visual=True):
    analytic = signal.hilbert(data)
    amplitude_envelope = np.abs(analytic)
    if visual:
        time = np.arange(0, len(data)/fs, 1/fs)
        fig, ax = plt.subplots()
        ax.plot(time, data, label = "signal")
        ax.plot(time, amplitude_envelope, label = "envelop")
        plt.legend()
        plt.show
    return amplitude_envelope

def ampPeak2Peak(data):


# def phaseDelay():
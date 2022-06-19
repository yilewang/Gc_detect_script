#!/bin/usr/python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import h5py
import sklearn
from typing import Union, List

def hdf5Reader(filename):
    """
    This function is for reading hdf5 file
    Input:
        filename: str
            the path of the h5 file
    Output:
        dataset: np array
            
    """
    with h5py.File(filename, "r") as f:
    ### get key of h5 file
        key = list(f.keys())
        dset = f[key[0]][:]
        return dset


# def visualization():




def psd(data, samplinginterval, visual=False, xlim=100., *args, **kwargs):
    """
    This function is for power spectrum density analysis
    Input:
        data: list or array
            LFPs single channel signal
        sampling interval: int or float
            sampling interval is the reciprocal of sampling frequency (1/fs)
        visual: boole, default is False

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
        data: list or np.array
            1-d array
        fs: int or float
            frequency (sampling rate)
        cut_off_low: int or float
            the low threshold
        cut_off_high: int or float
            the high threshold
        width: int or float
            the time windows for filtering
        ripple_db: int or float

    Return:
        filtered data: list or np.array

        N: int or float

        delay: int or float
            (for plotting)
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


def freqCount(data: Union[List[float], np.ndarray], prominence:Union[int, float], fs:float, normalization = False, filter=False, highpass = 2., lowpass = 10.,visual = False, figsize=None, dpi=None, *args, **kwargs) -> float:
    """
    A function designed to do spike counting.
    Input:
        data: list or np array
            the 1-D array signal
        prominence: int or float
            used in signal.find_peaks function, how salient the peaks are;
        fs: int or float
            sampling frequency;
        normalization: boole, default False
            normalize data by mean
        filter: boole, default False
            apply FIR filter or not;
        highpass: int or float
            if filter = True, the low threshold;
        lowpass: int or float
            if filter = True, the high cutoff threshold;
        visual: boole, default False
            the visualization;
        figsize: tuple
            the size of the plot
        dpi: default = None
            if visual = True, the pic's resolution;
    Output:
        if filter applied:
            (number of filtered signal, number of raw signal)
        else:
            (number of raw signal)
    """
    data = np.array(data)
    if normalization:
        data -= np.mean(data)
    time = np.arange(0, len(data)/fs, 1/fs)
    spikesdata, _ = signal.find_peaks(data, prominence = prominence)

    if filter:
        postfilter, N, delay= fir_bandpass(data, fs, highpass, lowpass)
        spikesfiltered, _ = signal.find_peaks(postfilter, prominence = prominence)
    # visualization
    if visual:
        fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
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


def ampCount(data:Union[List[float], np.ndarray], fs, prominence = None, threshold = None, normalization = None, mode = "peak2xais", visual=False) -> np.ndarray:
    """
    Input:
        data: 1-d list or np.array
            The single channel LFPs signal
        fs: int or float
            the sampling frequency
        prominence: int or float

        threshold: int or float

        normalization: boole, default is True
            normalization provided in this function is mean method.
        mode: str, "peak2valley", or "peak2xais", or "hilbert"
            different methods to calculate the amplitude:
            "peak2valley" mode:
                calculates the amplitude based on the maxmium distance between valley and spike within each cycle.
            "peak2xais" mode:
                calculate the absolute amplitude from spike to xais.
            "hilbert" mode:
                apply hilbert transformation to the signal. Warning: hilbert method only works in partial conditions.
        visual: boole, True or False
            display matplotlib plot.
    Output:
        Amplitude_data: float
            the average of amplitude across all cycles. The result depends on what mode is used in function. 
    """
    if mode in ["peak2valley", 'p2v']:
        valley, _ = signal.find_peaks(-data, prominence=prominence, threshold=threshold)

    
    elif mode in ["peak2axis", 'p20']:
        if normalization:
            data = data - np.mean(data)
        spikes, _ = signal.find_peaks(data, prominence=prominence, threshold=threshold)
        return np.mean(data[spikes])

    elif mode in ["hilbert","h"]:
        analytic = signal.hilbert(data)
        amplitude_envelope = np.abs(analytic)
        amplitude_data = np.mean(amplitude_envelope)
        if visual:
            time = np.arange(0, len(data)/fs, 1/fs)
            fig, ax = plt.subplots()
            ax.plot(time, data, label = "signal")
            ax.plot(time, amplitude_envelope, label = "envelop")
            plt.legend()
            plt.show
    else:
        raise ValueError("Invalid mode. Expected one of: %s" % mode)
    return amplitude_data




# def phaseDelay():
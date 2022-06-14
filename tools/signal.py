#!/bin/usr/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


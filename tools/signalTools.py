#!/bin/usr/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import h5py
from typing import Union, List




class signalToolkit:
    def __init__(self, filename, fs, caseid=None, group=None) -> None:
        self.filename = filename
        self.fs = fs
        self.samplinginterval = 1/fs
        self.casid = caseid
        self.group = group

    def hdf5Reader(self):
        """
        This function is for reading hdf5 file
        Parameters:
        -----------------
            filename: str
                the path of the h5 file
        Returns:
        -----------------
            dataset: np array
                
        """
        with h5py.File(self.filename, "r") as f:
        ### get key of h5 file
            key = list(f.keys())
            self.dset = f[key[0]][:]
            return self.dset



    def fir_bandpass(self, data, cut_off_low, cut_off_high, width=2.0, ripple_db=10.0):
        """
        The FIR bandpass filter
        Parameters:
        --------------------
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

        Returns:
        ---------------------
            filtered data: list or np.array

            N: int or float

            delay: int or float
                (for plotting)
            when plot, align the axis by `plt.plot(time[N-1:]-delay, filtered_data[N-1:])`
        """
        nyq_rate = self.fs / 2.0
        wid = width/nyq_rate
        N, beta = signal.kaiserord(ripple_db, wid)
        taps = signal.firwin(N, cutoff = [cut_off_low, cut_off_high],
                    window = 'hamming', pass_zero = False, fs=self.fs)
        filtered_signal = signal.lfilter(taps, 1.0, data)
        delay = 0.5 * (N-1) / self.fs
        return filtered_signal, N, delay

    

    def signalpreprocessing(self, channelNum, filter=True, low=None, high=None, spikesDetection = True):
        self.hdf5Reader()
        self.signal = self.dset[:,channelNum]
        if filter:
            self.filtered, self.N, self.delay = self.fir_bandpass(self.signal, low, high)
            if spikesDetection:
                self.peaks, _ = signal.find_peaks(self.filtered)
            return self.filtered, self.peaks
        if spikesDetection:
            self.peaks, _ = signal.find_peaks(self.signal)
            return self.signal, self.peaks

    def visual(pltFunc):
        def addFigAxes(self, figsize=None, digit=111, *args, **kwds):
            fig = plt.figure(figsize)
            axes = fig.add_subplot(digit)
            return pltFunc(self, axes, *args, **kwds)
        return addFigAxes
    
    @visual
    def psd(self, axes, visual=False, xlim=100., *args, **kwargs):
        """
        This function is for power spectrum density analysis
        
        Parameters:
        ---------------------
            data: list or array
                LFPs single channel signal
            sampling interval: int or float
                sampling interval is the reciprocal of sampling frequency (1/fs)
            visual: boole, default is False

        Returns:
        ----------------------
            Freq axis, PSD of the signal
        """
        total = len(self.signal)
        duration = total * self.samplinginterval
        fourierSignal = np.fft.fft(np.array(self.signal) - np.array(self.signal).mean())
        spectrum = 2 * (self.samplinginterval) ** 2 / duration * (fourierSignal * fourierSignal.conj())
        spectrum = spectrum[:int(len(np.array(self.signal)) / 2)]
        time_all = 1 / duration
        fNQ = 1/self.samplinginterval/2 # Nyquist frequency
        faxis = np.arange(0, fNQ, time_all) # frequency axis
        if visual:
            #fig, axs = plt.subplots()
            axes.plot(faxis, spectrum.real, color='r', label = 'PSD Results', *args, **kwargs)
            axes.legend()
            axes.set_xlim([0, xlim])
            plt.show()
        return faxis, spectrum.real






    def freqCount(self, data: Union[List[float], np.ndarray], prominence:Union[int, float], fs:float, normalization = False, filter=False, highpass = 2., lowpass = 10.,visual = False, figsize=None, dpi=None, *args, **kwargs) -> float:
        """
        A function designed to do spike counting.
        Parameters:
        ------------------
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
        Returns:
        -------------------
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
            postfilter, N, delay= self.fir_bandpass(data, fs, highpass, lowpass)
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


    def ampCount(self, data:Union[List[float], np.ndarray], fs, prominence = None, threshold = None, width = None, normalization = None, mode = "peak2xais", ampType='proportional', visual=False) -> np.ndarray:
        """
        Parameters:
        ---------------
            data: 1-d list or np.array
                The single channel LFPs signal
            fs: int or float
                the sampling frequency
            prominence: int or float

            threshold: int or float

            width: int or float

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
        Returns:
        --------------
            Amplitude_data: float
                the average of amplitude across all cycles. The result depends on what mode is used in function. 
        """
        if mode in ["peak2valley", 'p2v']:
            spikes, _ = signal.find_peaks(data, prominence=prominence)
            valley, _ = signal.find_peaks(-data, prominence=prominence, threshold=threshold, width=width)
            cycleSpikesMean = []
            for one in range(len(valley)-1):
                cycleSpikes = spikes[spikes > valley[one] and spikes < valley[one+1]]
                print(cycleSpikes)
                cycleSpikesMean.append(np.mean(cycleSpikes))
            return np.mean(cycleSpikesMean)

        elif mode in ["peak2axis", 'p20']:
            if normalization:
                data = data - np.mean(data)
            spikes, _ = signal.find_peaks(data, prominence=prominence, threshold=threshold)
            return np.mean(data[spikes])

        elif mode in ["hilbert","h"]:
            analytic = signal.hilbert(data)
            amplitude_envelope = np.abs(analytic)
            if visual:
                time = np.arange(0, len(data)/fs, 1/fs)
                fig, ax = plt.subplots()
                ax.plot(time, data, label = "signal")
                ax.plot(time, amplitude_envelope, label = "envelop")
                plt.legend()
                plt.show
            return np.mean(amplitude_envelope)
        else:
            raise ValueError("Invalid mode. Expected one of: %s" % mode)


    def phaseDelay(self, channelNum1, channelNum2, mode = "spikesInterval"):
        peakslist1, data1 = self.signalpreprocessing(channelNum1)
        peakslist2, data2 = self.signalpreprocessing(channelNum2)
        # if mode in ["spikeInterval", "SI"]:



        # elif mode in ["instaPhase", "IP"]:
        
        # else:
        #     raise ValueError("Invalid mode. Expected one of: %s" % mode)

#!/bin/usr/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import h5py
from typing import Union, List, Optional







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
        if cut_off_low - cut_off_high >=0:
            raise ValueError("Low pass value needs to be larger than High pass value.")
        nyq_rate = self.fs / 2.0
        wid = width/nyq_rate
        N, beta = signal.kaiserord(ripple_db, wid)
        taps = signal.firwin(N, cutoff = [cut_off_low, cut_off_high],
                    window = 'hamming', pass_zero = False, fs=self.fs)
        filtered_signal = signal.lfilter(taps, 1.0, data)
        delay = 0.5 * (N-1) / self.fs
        return filtered_signal, N, delay

    

    def signalpreprocessing(self, data=None, channelNum=None, filter=True, low=None, high=None, normalization = True):
        if data is None:
            self.hdf5Reader()
            self.signal = self.dset[:,channelNum]
            data = self.signal
        self.time = np.arange(0, len(data)/self.fs, 1/self.fs)
        if filter and normalization:
            afterFiltered, N, delay = self.fir_bandpass(data, low, high)
            afterFiltered = afterFiltered - np.mean(afterFiltered)
            return afterFiltered, N, delay
        elif filter and not normalization:
            afterFiltered, N, delay = self.fir_bandpass(data, low, high)
            return afterFiltered, N, delay
        elif normalization and not filter:
            data = data - np.mean(data)
            return data
        else:
            return data


    def peaksValleys(self, data, spikesparas:Optional[dict] = None, valleysparas:Optional[dict] = None):
        """
        function support to customize the `find_peaks` function to generate spikes and valleys data list
        Parameters
        ----------------------------
            data: 1-D list or np.ndarray
                LFP channel signal
            spikesparas: dict
                parameters such as prominence, width, or threshold
        

        Return
        -----------------------------
            spikeslist: list
            valleyslist: list
        
        """
        spikeslist, _ = signal.find_peaks(data, **spikesparas)
        valleyslist, _ = signal.find_peaks(-data, **valleysparas)
        return spikeslist, valleyslist
        

    def panel(pltFunc):
        def addFigAxes(self, figsize=None, *args, **kwds):
            fig = plt.figure(figsize)
            return pltFunc(self, fig, *args, **kwds)
        return addFigAxes
    
    @panel
    def signal_AF(self, fig, time, data, spikeslist, valleylist, N, delay, afterFiltered, spikeslistAF, valleyslistAF,digit=111, newkwargs={}):
        axes = fig.add_subplot(digit)
        axes.plot(time, data, label = "signal")
        axes.plot(spikeslist/self.fs, data[spikeslist], '+', label = "signal spikes")
        axes.plot(time[N-1:]-delay, afterFiltered[N-1:], label = "filtered signal")
        if len(spikeslistAF) > 0:
            axes.plot(spikeslistAF[spikeslistAF > N-1]/self.fs - delay, spikeslistAF[spikeslistAF > N-1],'x', label = "filtered spikes")
        axes.plot(**newkwargs)
        axes.legend()
        plt.show()
    
    @panel
    def psd(self, fig, digit=None, data=None, visual=False, filtered=True, xlim=100., *args, **kwargs):
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
        if data is None:
            data = self.signal
        if filtered:
            data = self.filtered
        total = len(data)
        duration = total * self.samplinginterval
        fourierSignal = np.fft.fft(np.array(data) - np.array(data).mean())
        spectrum = 2 * (self.samplinginterval) ** 2 / duration * (fourierSignal * fourierSignal.conj())
        spectrum = spectrum[:int(len(np.array(data)) / 2)]
        time_all = 1 / duration
        fNQ = 1/self.samplinginterval/2 # Nyquist frequency
        faxis = np.arange(0, fNQ, time_all) # frequency axis
        if visual:
            axes = fig.add_subplot(digit)
            axes.plot(faxis, spectrum.real, color='r', label = 'PSD Results', *args, **kwargs)
            axes.legend()
            axes.set_xlim([0, xlim])
            plt.show()
        return faxis, spectrum.real


    def freqCount(self, spikeslist) -> float:
        """
        A function designed to do spike counting.
        Parameters:
        ------------------
        Returns:
        -------------------
            (number of raw signal)
        """
        return len(spikeslist)

    @panel
    def ampCount(self, data, spikeslist, valleyslist, mode = "peak2xais", ampType='proportional', visual=False, fig=None, digit=None, spikesparas:Optional[dict] = None, valleysparas:Optional[dict] = None) -> np.ndarray:
        """
        Parameters:
        ---------------
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
        if data is None:
            data = self.signal
            spikeslist, valleyslist = self.peaksValleys(self.signal, spikesparas, valleysparas)
        if mode in ["peak2valley", 'p2v']:
            cycleSpikesMean = []
            for one in range(len(valleyslist)-1):
                cycleSpikes = spikeslist[spikeslist > valleyslist[one] and spikeslist < valleyslist[one+1]]
                cycleSpikesMean.append(np.mean(data[cycleSpikes]))
            return np.mean(cycleSpikesMean)

        elif mode in ["peak2axis", 'p20']:
            return np.mean(data[spikeslist])

        elif mode in ["hilbert","h"]:
            analytic = signal.hilbert(self.signal)
            amplitude_envelope = np.abs(analytic)
            if visual:
                time = np.arange(0, len(data)/self.fs, 1/self.fs)
                axes = fig.add_subplot(digit)
                axes.plot(time, data, label = "signal")
                axes.plot(time, amplitude_envelope, label = "envelop")
                plt.legend()
                plt.show
            return np.mean(amplitude_envelope)
        else:
            raise ValueError("Invalid mode. Expected one of: %s" % mode)

    @panel
    def phaseDelay(self, data1=None, data2=None, spikeslist1 = None, valleyslist1=None, spikeslist2 = None, valleyslist2 = None, channelNum1:Optional[int] = None, channelNum2:Optional[int] = None, preproparas:Optional[dict] = None, spikesparas:Optional[dict] = None, valleysparas:Optional[dict] = None, mode = "spikesInterval"):
        
        if None in (data1, data2, spikeslist1, valleyslist1, spikeslist2, valleyslist2):
            data1= self.signalpreprocessing(channelNum1, **preproparas)
            spikeslist1, valleyslist1 = self.peaksValleys(data1, spikesparas, valleysparas)
            data2= self.signalpreprocessing(channelNum2, **preproparas)
            spikeslist2, valleyslist2 = self.peaksValleys(data2, spikesparas,valleysparas)
        

        if mode in ["spikeInterval", "SI"]:
            

        elif mode in ["instaPhase", "IP"]:
        
        else:
            raise ValueError("Invalid mode. Expected one of: %s" % mode)

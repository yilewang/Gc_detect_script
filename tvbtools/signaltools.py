#!/bin/usr/python

from cmath import phase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import h5py
from typing import Type, Union, List, Optional
import neurokit2 as nk
from functools import wraps
import os
import io

class SignalToolkit:
    def __init__(self, filename=None, fs=None, caseid=None, group=None) -> None:
        """
        The initial parameters required for the data analysis
        Parameters:
        ----------------
            filename:str
                path of the file
            fs: float
                the sampling frequency
            caseid:str
                the subject's case ID number
            group:str
                group information of the subject
        """
        self.filename = filename
        self.fs = fs
        if self.fs is None:
            pass
        else:
            self.sampling_interval = 1/fs
        self.casid = caseid
        self.group = group




    def txt_reader(self) -> np.ndarray:
        openSC = open(self.filename,"r")
        lines = openSC.read()
        df = pd.read_csv(io.StringIO(lines), sep='\t', header=None, index_col=None, engine="python")
        return df.to_numpy()

    def csv_reader(self) -> np.ndarray:
        df = pd.read_csv(self.filename, index_col=0)
        return df.to_numpy()

    def excel_reader(self) -> np.ndarray:
        df = pd.read_excel(self.filename, index_col=0)
        return df.to_numpy()
    
    def mat_reader(self) -> dict:
        mat = scipy.io.loadmat(self.filename)
        return mat

    def hdf5_reader(self):
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
            dset = f[key[0]][:]
            return dset

    def data_reader(self):
        _, file_extension = os.path.splitext(self.filename)
        if file_extension in [".csv"]:
            data = self.csv_reader()
        elif file_extension in [".xlsx"]:
            data = self.excel_reader()
        elif file_extension in [".txt"]:
            data = self.txt_reader()
        elif file_extension in [".hdf5", ".h5"]:
            data = self.hdf5_reader()
        elif file_extension in [".mat"]:
            data = self.mat_reader()
        else:
            raise TypeError("The data type is not supported by this reader. Please contact yile.wang@utdallas.edu to add support to other datatypes. Thanks")
        return data

    def fir_bandpass(self, data, cut_off_low, cut_off_high, fs=None, width=2.0, ripple_db=10.0):
        """
        The FIR bandpass filter
        Parameters:
        --------------------
            data: list or np.array
                1-d array
            cut_off_low: int or float
                the high pass threshold
            cut_off_high: int or float
                the low pass threshold
            width: int or float
                the time windows for filtering
            ripple_db: int or float
                Ripples are the fluctuations (measured in dB) in the pass band, or stop band, of a filter's frequency magnitude response curve
        Returns:
        ---------------------
            filtered data: list or np.array
                data after filtered
            N: int or float
                The truncation signal points
            delay: int or float
                (for plotting) when plot, align the axis by `plt.plot(time[N-1:]-delay, filtered_data[N-1:])`
        """
        if fs is None:
            fs = self.fs
        if cut_off_low - cut_off_high >=0:
            raise ValueError("Low pass value needs to be larger than High pass value.")
        nyq_rate = fs / 2.0
        wid = width/nyq_rate
        N, beta = signal.kaiserord(ripple_db, wid)
        taps = signal.firwin(N, cutoff = [cut_off_low, cut_off_high],
                    window = 'hamming', pass_zero = False, fs=fs)
        filtered_signal = signal.lfilter(taps, 1.0, data)
        delay = 0.5 * (N-1) / fs
        return filtered_signal, N, delay



    def signal_preprocessing(self, data, truncate = 0, filter=True, low=None, high=None, normalization = False):
        """
        The function to help doing FIR bandpass filter and normalization for the signal
        Parameters:
        -----------------
            data:list or np.ndarray
                the signal
            filter: boole
                filter or not
            low (optional):float
                if filter is True, you have to provide high pass parameter
            high (optional):float
                if filter is True, you have to provide low pass parameter
            normalization: boole
                to choose if you want to normalize your data or not
        Returns:
        -------------------
            if filter is True:
                return data_after_filtered, N and delay
            else:
                data itself
        """
        if truncate > 0:
            ind = int(self.fs*truncate)
            data = data[ind:]
        elif truncate < 0:
            raise ValueError("truncate number must be over 0")
        self.time = np.arange(0, len(data)/self.fs, 1/self.fs)
        if filter and normalization:
            after_filtered, N, delay = self.fir_bandpass(data, low, high)
            after_filtered = after_filtered - np.mean(after_filtered)
            return after_filtered, N, delay
        elif filter and not normalization:
            after_filtered, N, delay = self.fir_bandpass(data, low, high)
            return after_filtered, N, delay
        elif normalization and not filter:
            data = data - np.mean(data)
            return data
        else:
            return data

    def signal_package(self, data, channel_num, label, low, high, normalization = True,spikesparas=None, valleysparas=None, spikesparas_af=None, valleysparas_af = None,truncate = 0):
        """
        a function to return signal preprocessing info in a pack
        Parameters:
        -----------------------
            dset:np.ndarray
                the dataset return from `self.hdf5_reader`
            channel_num:int
                the channel number to be analyzed
            label:str
                the name of the brain region
            Normalization:boole
                to normalize data
        Returns:
        -----------------------
            packdict:dict
                A dict including all necessary information
                {"data":roi, 
                "after_filtered":roi_af, 
                "spikeslist":spikeslist, 
                "spikeslist_af":spikeslist_af, 
                "valleyslist":valleyslist, 
                "valleyslist_af":vallyeslist_af,
                "N":N, 
                "delay":delay, 
                "label":label}
        """
        roi = self.signal_preprocessing(data[:,channel_num], filter=False, normalization=normalization, truncate=truncate)
        roi_af, N, delay = self.signal_preprocessing(roi, truncate = 0, filter = True, normalization = normalization, low=low, high=high)
        spikeslist, valleyslist = self.peaks_valleys(roi, spikesparas, valleysparas)
        spikeslist_af, valleyslist_af = self.peaks_valleys(roi_af, spikesparas_af, valleysparas_af)
        packdict = {"data":roi, "after_filtered":roi_af, "spikeslist":spikeslist, "spikeslist_af":spikeslist_af, "valleyslist":valleyslist_af, "valleyslist_af":valleyslist_af, "N":N, "delay":delay, "label":label}
        return packdict
            


    def peaks_valleys(self, data, spikesparas:Optional[dict] = None, valleysparas:Optional[dict] = None):
        """
        function support to customize the `find_peaks` function to generate spikes and valleys data list
        Parameters
        ----------------------------
            data: 1-D list or np.ndarray
                LFP channel signal
            spikesparas: dict
                parameters such as prominence, width, or threshold
            valleysparas: dict
                parameters such as prominence, width, or threshold
        Return
        -----------------------------
            spikeslist: list
            valleyslist: list
        """
        spikeslist, _ = signal.find_peaks(data, **spikesparas)
        valleyslist, _ = signal.find_peaks(-data, **valleysparas)
        return spikeslist, valleyslist
        

    # def panel(func):
    #     """
    #     A python decorator to provide plot panel
    #     """
    #     def addFigAxes(self, figsize=(15,5), *args, **kwargs):
    #         fig = plt.figure(figsize=figsize)
    #         return func(self, fig, *args, **kwargs)
    #     return addFigAxes
    

    #@panel
    def signal_af(self, axes, data=None, spikeslist=None, valleyslist=None, N=None, delay=None, after_filtered=None, spikeslist_af=None, valleyslist_af = None, digit=111, time=None, label=None, **kwargs):
        """
        A plotting function to visualize signal, signal after filtered, spikes, spikes after filtered, valleys in one plot
        Parameters:
        ---------------
            fig:function
                inherit from decorator
            data:list or np.ndarray
                the signal
            spikeslist:list
                The collection of spikes points
            valleyslist:list
                the collection of vallyes points
            N: int or float
                truncate data points number in the beginning of the signal. Inherited from `fir_bandpass` function
            delay: int or float
                time delay from `fir_bandpass` function
            after_filtered: list or np.ndarray
                signal after filtered
            spikeslist_af: list
                the spikes points of the filtered signal
            digit:int, default 111
                add subplot to the panel
        """
        if time is None:
            time = self.time
        if axes is None:
            fig = plt.figure(figsize=(15,5))
            axes = fig.add_subplot(111)
        axes.plot(time, data, label = "signal")
        axes.plot(spikeslist/self.fs, data[spikeslist], '+', label = "signal spikes")
        axes.plot(valleyslist_af[valleyslist_af > N-1]/self.fs-delay, after_filtered[valleyslist_af[valleyslist_af > N-1]], 'o', label = "filtered valleys")
        axes.plot(time[N-1:]-delay, after_filtered[N-1:], label = "filtered signal")
        if len(spikeslist_af) > 0:
            axes.plot(spikeslist_af[spikeslist_af > N-1]/self.fs - delay, after_filtered[spikeslist_af[spikeslist_af > N-1]],'x', label = "filtered spikes")
        return axes

    def psd(self, data, sampling_interval = None, visual=False, xlim=100.,axes=None, fNQ = None, normalization = True, *args, **kwargs):
        """
        This function is for power spectrum density analysis
        
        Parameters:
        ---------------------
            data: list or array
                LFPs single channel signal
            sampling interval: int or float
                sampling interval is the reciprocal of sampling frequency (1/fs)
            visual: boole, default is False
                execute the visualization or not
            xlim: int or float
                for better visualize frequency band in plot
        Returns:
        ----------------------
            Freq axis, PSD of the signal
        """
        if sampling_interval is None:
            sampling_interval = self.sampling_interval
        fs = 1/sampling_interval
        if fNQ is None:
            fNQ = int(fs/2)
        data = np.array(data)
        if normalization:
            data = data-np.mean(data)
        fourier = np.fft.fft(data, n=fNQ)
        fourier = fourier[0:int(fNQ/2)]
        faxis  = np.arange(int(fNQ/2)) * (fs/fNQ)

        if visual:
            if axes is None:
                fig = plt.figure(figsize=(15,5))
                axes = fig.add_subplot(111)
            axes.plot(faxis, np.abs(fourier**2), *args, **kwargs)
            axes.set_xlim([0, xlim])
        return faxis, np.abs(fourier)**2

    def freq_count(self,spikeslist, data=None,  visual=False, axes=None) -> float:
        """
        A function designed to do spike counting.
        Parameters:
        ------------------
            spikeslist:list or np.ndarray
                the spikeslist, can be generated by `peaks_valleys` function
        Returns:
        -------------------
            number of spikes in signal:int
        """
        if visual:
            if axes is None:
                fig = plt.figure(figsize=(15,5))
                axes = fig.add_subplot(111)
            axes.plot(data)
            for one in spikeslist:
                axes.vlines(one, ymin=0, ymax=data[one], colors = 'purple')
            axes.set_title("frequency spikes count")
        return len(spikeslist)

    @staticmethod
    def range_peaks(spikeslist, valleyslist, init, end):
            spikeslist = np.array(spikeslist)
            valleyslist= np.array(valleyslist)
            return spikeslist[np.where(np.logical_and(spikeslist>=init, spikeslist<=end))]

    def amp_count(self, data, spikeslist, valleyslist, mode = "peak2xais", visual=False, spikeslist_af:Optional[list] = None, after_filtered=None, N=None, delay=None, axes = None, **kwargs):
        """
        Parameters:
        ---------------
            data:list or np.ndarray
                signal
            spikeslist:list or np.ndarray
                spikes of the signal
            valleyslist:list or np.ndarray
                valleys points of the signal
            mode: str, "peak2valley", or "peak2xais", or "hilbert"
                different methods to calculate the amplitude:
                "peak2valley" mode:str
                    calculates the amplitude based on the maxmium distance between valley and spike within each cycle.
                "peak2xais" mode:str
                    calculate the absolute amplitude from spike to xais.
                "ampProportional" mode:str
                    calculate the proportional info between signal amplitude and signal after_filtered amplitude.
                "hilbert" mode:
                    apply hilbert transformation to the signal. Warning: hilbert method only works in partial conditions.
            visual: boole, True or False
                display matplotlib plot.
            fig:function
                inherited from @panel
            digit:int, default is 111
                for adding subplot
        Returns:
        --------------
            Amplitude_data: list
                the amplitude of all cycles. The result depends on what mode is used in function. 
        """
        
        if mode in ["peak2valley", 'p2v']:
            cycle_spikes_mean = []
            _init = 0
            if len(valleyslist) <= 0:
                return 0
            else:
                for one in valleyslist:
                    cycle_spikes = self.range_peaks(spikeslist=spikeslist, valleyslist=valleyslist, init=_init, end=one)
                    if len(cycle_spikes) >0:
                        peak2val = np.mean(data[cycle_spikes])-np.mean([data[_init], data[one]])
                        cycle_spikes_mean.append(peak2val)
                    _init = one
                if visual:
                    if axes is None:
                        fig = plt.figure(figsize=(15,5))
                        axes = fig.add_subplot(111)
                    axes.plot(data)
                    for one in cycle_spikes_mean:
                        axes.vlines(one, ymin=0, ymax=data[one], colors = 'purple')
                    plt.show()
                return np.mean(cycle_spikes_mean)

        elif mode in ["peak2axis", 'p20']:
            cycle_spikes_mean = []
            _init = 0
            if len(valleyslist) <= 0:
                return 0
            else:
                for one in valleyslist:
                    cycle_spikes = self.range_peaks(spikeslist=spikeslist, valleyslist=valleyslist, init=_init, end=one)
                    if len(cycle_spikes) >0:
                        peak2val = np.mean(data[cycle_spikes])
                        cycle_spikes_mean.append(peak2val)
                    _init = one
                if visual:
                    if axes is None:
                        fig = plt.figure(figsize=(15,5))
                        axes = fig.add_subplot(111)
                    axes.plot(data)
                    for one in spikeslist:
                        axes.vlines(one, ymin=0, ymax=data[one], colors = 'purple')
                    plt.show()
                return np.mean(cycle_spikes_mean)
        
        elif mode in ["ampProportional", "ap", "AP"]:
            amp_upper_pro = []
            amp_lower_pro = []
            _init = 0
            if len(valleyslist) <= 0:
                return 0, 0
            else:
                for one in valleyslist:
                    raw_spikes = self.range_peaks(spikeslist=spikeslist, valleyslist=valleyslist, init=_init, end = one)
                    spikes_af = self.range_peaks(spikeslist=spikeslist_af, valleyslist=valleyslist, init=_init, end = one)
                    if len(raw_spikes) and len(spikes_af) >0:
                        peak2val = np.mean(data[raw_spikes]-np.mean([data[_init], data[one]]))
                        peak2val_af = np.mean(data[spikes_af]-np.mean([data[_init], data[one]]))
                        upper_pro = (peak2val - peak2val_af)/peak2val
                        lower_pro = peak2val_af / peak2val
                        amp_upper_pro.append(upper_pro)
                        amp_lower_pro.append(lower_pro)
                    _init = one
                if visual:
                    if axes is None:
                        fig = plt.figure(figsize=(15,5))
                        axes = fig.add_subplot(111)
                    axes.plot(data)
                    axes.plot(after_filtered)
                    for one,two in zip(amp_upper_pro,amp_lower_pro):
                        axes.vlines(one, ymin=0, ymax=data[one], colors = 'purple')
                        axes.vlines(two[two>N-1]-delay*self.fs, ymin=0, ymax=after_filtered[two[two>N-1]], colors = 'green')
                    plt.show()
                return np.mean(amp_upper_pro), np.mean(amp_lower_pro)

        elif mode in ["hilbert","h"]:
            analytic = signal.hilbert(self.signal)
            amplitude_envelope = np.abs(analytic)
            if visual:
                time = np.arange(0, len(data)/self.fs, 1/self.fs)
                if axes is None:
                    fig = plt.figure(figsize=(15,5))
                    axes = fig.add_subplot(111)
                axes.plot(time, data, label = "signal")
                axes.plot(time, amplitude_envelope, label = "envelop")
                plt.legend()
                plt.show
            return np.mean(amplitude_envelope)
        else:
            raise ValueError("Invalid mode. Expected one of: %s" % mode)
    

    def phase_delay(self, data1, data2, spikeslist1, valleyslist1, spikeslist2, valleyslist2, mode = "spikesInterval"):
        """
        A function used to calculate the delay between two signals
        Parameters:
        -----------------------
            data1:list, np.ndarray
                signal 1
            dara2:list, np.ndarray
                signal 2
            spikeslist1:list, np.ndarray
                the spikes points of the signal 1
            spikeslist2:list, np.ndarray
                the spikes points of the signal 2
            valleyslist1:list, np.ndarray
                the valleys points of the signal 1
            valleyslist2:list, np.ndarray
                the valleys points of the signal 2
            mode:str, the options are `spikesInterval`, `instaPhase`, and `windowsPhase`
                spikesInterval aims to calculate the 
        Return:
        -----------------------
            delay list
        """
        if len(spikeslist1)<=0 or len(spikeslist2) <= 0:
            return 'N/A'
        def first_spike(spikeslist, valleyslist):
            cycle1spikes = []
            _init=0
            for one in valleyslist:
                try:
                    firstspike = self.range_peaks(spikeslist=spikeslist, valleyslist=valleyslist, init=_init, end = one)[0]
                    cycle1spikes.append(firstspike)
                except IndexError:
                    pass
                _init = one
            return cycle1spikes

        if mode in ["spikeInterval", "SI"]:
            data1spikes = first_spike(spikeslist1, valleyslist1)
            data2spikes = first_spike(spikeslist2, valleyslist2)
            # get shorter list length
            spikeslen = min(len(data1spikes), len(data2spikes))
            delaylist = []
            for one in range(spikeslen):
                if data1spikes[0] > data2spikes[0]:
                    diff = data2spikes[one] - data1spikes[one]
                    delaylist.append(diff)
                else:
                    diff = data1spikes[one] - data2spikes[one]
                    delaylist.append(diff)
            return np.mean(delaylist)/self.fs
            

        elif mode in ["instaPhase", "IP"]:
            phase1 = np.angle(signal.hilbert(data1),deg=False)
            phase2 = np.angle(signal.hilbert(data2),deg=False)
            synchrony = 1 - np.sin(np.abs(phase1 - phase2) / 2)
            return np.mean(synchrony)

        elif mode in ["windowsPhase", "WP"]:
            synchrony = nk.signal_synchrony(data1, data2, method="correlation", window_size=50)
            return np.mean(synchrony)
        
        else:
            raise ValueError("Invalid mode. Expected one of: %s" % mode)
    
    def phase_locking(self, data1,data2, visual=False, axes = None):
        """
        A function to calculate phase locking value. The mathematical formula is 
        $$PLV_t = \frac{1}{N}\abs{\sum_{n=1}{N}exp(\mathbf j \theta (t, n)}$$
        Parameters:
        ----------------------------
            data1:list or np.ndarray
                signal 1
            data2:list or np.ndarray
                siganl 2
        Returns:
        ----------------------------
            phase locking value
        """

        h1=signal.hilbert(data1)
        h2=signal.hilbert(data2)
        # pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
        #            np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
        # phase = np.angle(pdt)
        diff = h1 - h2
        plv = np.exp(np.array([complex(0, diff[i]) for i in range(len(diff))]))
        plv = np.abs(plv)
        if visual:
            if axes is None:
                fig = plt.figure(figsize=(15,5))
                axes = fig.add_subplot(111)
            axes.plot(plv)
            plt.show()
        return np.mean(plv)


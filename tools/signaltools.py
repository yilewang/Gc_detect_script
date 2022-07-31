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
import seaborn as sns

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

    def time_maker(self, data, fs=None):
        if fs is None:
            fs = self.fs
        timeaxis = np.arange(0, len(data)/fs, 1/fs)
        return timeaxis

    @staticmethod
    def hamming_filter(data, Wn, fNQ=81920/2, n=2048):
        a = signal.firwin(n, Wn, nyq=fNQ, pass_zero=False, window='hamming')
        filtered_data = signal.filtfilt(a, 1, data);   # ... and apply it to the data
        return filtered_data

    @staticmethod
    def sos_filter(data, win, fs, order=5):
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
            return sos
        
        sos = butter_bandpass(win[0],win[1], fs,order=order)
        y = signal.sosfiltfilt(sos, data)
        return y

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
            after_filtered= self.hamming_filter(data, [low, high], self.fs/2)
            after_filtered = after_filtered - np.mean(after_filtered)
            return after_filtered
        elif filter and not normalization:
            after_filtered= self.hamming_filter(data, [low, high], self.fs/2)
            return after_filtered
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
                "label":label}
        """
        roi = self.signal_preprocessing(data[:,channel_num], filter=False, normalization=normalization, truncate=truncate)
        roi_af= self.signal_preprocessing(roi, truncate = 0, filter = True, normalization = normalization, low=low, high=high)
        spikesparas['height']= roi_af
        spikeslist, valleyslist = self.peaks_valleys(roi, spikesparas, valleysparas)
        spikeslist_af, valleyslist_af = self.peaks_valleys(roi_af, spikesparas_af, valleysparas_af)
        packdict = {"data":roi, "after_filtered":roi_af, "spikeslist":spikeslist, "spikeslist_af":spikeslist_af, "valleyslist":valleyslist_af, "valleyslist_af":valleyslist_af, "label":label}
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
        axes.plot(valleyslist_af/self.fs, after_filtered[valleyslist_af], 'o', label = "filtered valleys")
        axes.plot(time, after_filtered, label = "filtered signal")
        if len(spikeslist_af) > 0:
            axes.plot(spikeslist_af/self.fs - delay, after_filtered[spikeslist_af],'x', label = "filtered spikes")
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
            if len(valleyslist) <= 0 or len(spikeslist) <= 0:
                return 0
            else:
                for one in valleyslist:
                    cycle_spikes = self.range_peaks(spikeslist=spikeslist, valleyslist=valleyslist, init=_init, end=one)
                    if len(cycle_spikes) >0:
                        peak2val = np.mean(data[cycle_spikes])-np.mean([data[_init], data[one]])
                        cycle_spikes_mean.append(peak2val)
                    else:
                        cycle_spikes_mean.append(0)
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
            if len(valleyslist) <= 0 or len(spikeslist) <= 0:
                return 0
            else:
                for one in valleyslist:
                    cycle_spikes = self.range_peaks(spikeslist=spikeslist, valleyslist=valleyslist, init=_init, end=one)
                    if len(cycle_spikes) >0:
                        peak2val = np.mean(data[cycle_spikes])
                        cycle_spikes_mean.append(peak2val)
                    else:
                        cycle_spikes_mean.append(0)
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
                    elif len(raw_spikes) <=0 and len(spikes_af) >0:
                        amp_upper_pro.append(0)
                        amp_lower_pro.append(1)
                    elif len(raw_spikes) <=0 and len(spikes_af) <=0:
                        amp_lower_pro.append(0)
                        amp_upper_pro.append(0)
                    _init = one
                if visual:
                    if axes is None:
                        fig = plt.figure(figsize=(15,5))
                        axes = fig.add_subplot(111)
                    axes.plot(data)
                    axes.plot(after_filtered)
                    for one,two in zip(amp_upper_pro,amp_lower_pro):
                        axes.vlines(one, ymin=0, ymax=data[one], colors = 'purple')
                        axes.vlines(two, ymin=0, ymax=after_filtered[two], colors = 'green')
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
            return 1.0
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
                diff = data2spikes[one] - data1spikes[one]
                delaylist.append(np.abs(diff))
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
        phase_y1=np.unwrap(np.angle(h1))
        phase_y2=np.unwrap(np.angle(h2))
        # pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
        #            np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
        # phase = np.angle(pdt)
        complex_phase_diff = np.exp(np.complex(0,1)*(phase_y1 - phase_y2))
        plv = np.abs(np.sum(complex_phase_diff))/len(phase_y1)
        return plv
        # plv = sum(np.exp(np.array([complex(0, diff[i]) for i in range(len(diff))])))
        # plv = np.abs(plv)
        # if visual:
        #     if axes is None:
        #         fig = plt.figure(figsize=(15,5))
        #         axes = fig.add_subplot(111)
        #     axes.plot(plv)
        #     plt.show()
        # return plv

    @staticmethod
    def lateral(data1, data2, abs=True):
        data1 = np.array(data1)
        data2 = np.array(data2)
        # if len(data1) != len(data2):
        #     raise ValueError("the two dataset should have exact same size")
        if data1+data2 == 0:
            return 0.5

        if abs:
            la = np.abs((data1 - data2))/(data1 + data2)
            return np.mean(la, dtype=np.float64)
        else:
            la = (data1 - data2)/(data1 + data2)
            return np.mean(la, dtype=np.float64)

    @staticmethod
    def psvplot(x,y, df, colors=None, axes=None):
        sns.violinplot(x=x, y=y, data=df, palette=colors, inner = None, width=0.7, bw=0.2, ax=axes)
        sns.stripplot( x=x, y=y, data=df, color='black', label="right", ax = axes)
        sns.pointplot(x=x, y=y, data=df, estimator=np.mean, color = 'red', ax=axes)
    

    # cross frequency coupling tools: phase amplitude coupling, based on tort, 2010 method
    @staticmethod
    def PAC(data, low_win, high_win, fs=81920, n_bins=36, visual=False):
        """
        Parameters:
        ---------------
            data:list or np.ndarray
                signal
            low_win:list
                start frequency, stop frequency
            high_win:list
                start frequency, stop frequency
        Returns:
        --------------
            Modulation Index (MI)
        """
        # unify data format
        data = np.array(data)
        # filtering data into high frequency band and low frequency band
        low_freq = SignalToolkit.sos_filter(data, low_win, fs=fs)
        high_freq = SignalToolkit.sos_filter(data, high_win, fs=fs)
        # hilbert transform
        # step 1, get the amplitude envelop of high freq
        h1=signal.hilbert(high_freq)
        amplitude_envelope = np.abs(h1)
        # step 2, get the phase information of low freq
        l1 = signal.hilbert(low_freq)
        phase_y1=np.degrees(np.angle(l1))
        # step 3 bin the phase
        binsize = 360/ n_bins
        phase_bins = np.arange(-180,180,binsize)
        amp_mean = np.zeros(len(phase_bins)) 
        for k in range(len(phase_bins)):
            phase_range = np.logical_and(phase_y1 >= phase_bins[k],
                                        phase_y1 < (phase_bins[k] + binsize))
            tmp_amp = amplitude_envelope[phase_range]
            amp_mean[k] = np.mean(tmp_amp)   
        # step 4, entropy method H
        p_j = amp_mean / np.sum(amp_mean)
        # cap_H = -np.sum(p_j * np.log(p_j))
        # # step 5, calculate the MI
        # MI = (np.log(len(phase_bins))-cap_H) / np.log(len(phase_bins))
        # kl_pu = np.log(len(phase_bins)) + np.sum(p_j * np.log(p_j))
        # MI = kl_pu/np.log(len(phase_bins))
        # print(f"Modulation Index = {MI}")
        if np.any(p_j == 0):
            p_j[p_j == 0] = np.finfo(float).eps

        H = -np.sum(p_j * np.log10(p_j))
        Hmax = np.log10(n_bins)
        KL = Hmax - H
        MI = KL / Hmax
        if visual:
            fig = plt.figure(figsize=(7,7))
            # graph 1
            axes1 = fig.add_subplot(221)
            axes1.set_title("distribution of the mean amplitude\n in each phase bin")
            axes1.bar(phase_bins, amp_mean)
            # graph 2
            axes2 = fig.add_subplot(222)
            axes2.set_title("raw plot with low and high \nfrequency bands signal")
            axes2.plot(data)
            axes2.plot(low_freq)
            axes2.plot(high_freq)
            # graph 3
            axes3 = fig.add_subplot(223)
            axes3.set_title("amplitude & phase")
            axes3.plot(phase_y1, amplitude_envelope)
            # graph 4
            axes4 = fig.add_subplot(224)
            axes4.set_title("phase of low frequency")
            axes4.plot(low_freq)
            axes4.plot(phase_y1)
            plt.show()
        return MI
        
    @staticmethod
    def PAC_comodulogram(data, low_paras, high_paras, fs, axes = None, **plot_keyargs):
        data = np.array(data)
        phase_x = np.arange(*low_paras)
        amplitude_y = np.arange(*high_paras)
        como_df = pd.DataFrame(index=phase_x, columns=amplitude_y)
        for xi, i in enumerate(phase_x):
            for yi, j in enumerate(amplitude_y):
                mi = SignalToolkit.PAC(data, [i, phase_x[xi]+low_paras[2]], [j, amplitude_y[yi]+high_paras[2]], fs=fs, visual = False)
                como_df.iloc[xi,yi] = mi
                como_df[como_df.columns[yi]] = como_df[como_df.columns[yi]].astype(float, errors = 'raise')
        como_df.columns += high_paras[2]/2
        como_df.index += low_paras[2]/2
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        sns.heatmap(como_df.transpose(), ax = axes, cmap="coolwarm", **plot_keyargs)
        axes.invert_yaxis()
        #_developing
        # old_ticks = axes.get_xticks()
        # new_ticks = np.linspace(np.min(old_ticks), np.max(old_ticks), len(phase_x))
        # axes.set_xticks(new_ticks)
        # axes.set_xticklabels()
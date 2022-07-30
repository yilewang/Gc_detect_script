#!/usr/bin/python

import numpy as np

import matplotlib.pyplot as plt











def PAC(data, low_win, high_win, fs, n_order_low = 256, n_order_high=2048, visual=False):
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

    # def sign_check(data):
    #     signs = np.sign(data)
    #     places = []
    #     for ii in range(1, len(signs)):
    #         if signs[ii] != signs[ii-1]:
    #             places.append(ii)
    #     return len(places)

    # unify data format
    data = np.array(data)
    # filtering data into high frequency band and low frequency band
    low_freq = SignalToolkit.hamming_filter(data, Wn=low_win, fNQ=fs/2, n=n_order_low)
    high_freq = SignalToolkit.hamming_filter(data, Wn=high_win, fNQ=fs/2, n=n_order_high)
    # hilbert transform
    # step 1, get the amplitude envelop of high freq
    h1=signal.hilbert(high_freq)
    amplitude_envelope = np.abs(h1)
    # step 2, get the phase information of low freq
    l1 = signal.hilbert(low_freq)
    phase_y1=np.angle(l1)
    # step 3 bin the phase
    phase_bins = np.arange(-np.pi,np.pi,0.1)
    amp_mean = np.zeros(np.size(phase_bins)-1)      
    phase_mean = np.zeros(np.size(phase_bins)-1)     
    for k in range(np.size(phase_bins)-1):   
        phase_low = min(phase_bins[k], phase_bins[k+1]) 
        phase_high = max(phase_bins[k], phase_bins[k+1]) 
        tmp_amp = amplitude_envelope[np.where(np.logical_and(phase_y1>=phase_low, phase_y1<=phase_high))]
        amp_mean[k] = np.mean(tmp_amp)   
        phase_mean[k] = np.mean([phase_low, phase_high]) 
    # step 4, entropy method H
    p_j = [p_j_single/np.sum(amp_mean) for p_j_single in amp_mean]
    cap_H = -np.sum(p_j * np.log(p_j))
    # step 5, calculate the MI
    MI = (np.log(len(phase_bins))-cap_H) / np.log(len(phase_bins))
    print(f"Modulation Index = {MI}")
    if visual:
        fig = plt.figure(figsize=(7,7))
        # graph 1
        axes1 = fig.add_subplot(221)
        axes1.set_title("distribution of the mean amplitude\n in each phase bin")
        axes1.bar(phase_bins[:-1], amp_mean)
        # graph 2
        axes2 = fig.add_subplot(222)
        axes2.set_title("raw plot with low and high \nfrequency bands signal")
        axes2.plot(data)
        axes2.plot(low_freq)
        axes2.plot(high_freq)
        # graph 3
        axes3 = fig.add_subplot(223)
        axes3.set_title("amplitude envelope\n of high frequency")
        axes3.plot(high_freq)
        axes3.plot(amplitude_envelope)
        # graph 4
        axes4 = fig.add_subplot(224)
        axes4.set_title("phase of low frequency")
        axes4.plot(low_freq)
        axes4.plot(phase_y1)
        plt.show()
    return MI


data = df_left
low_win = [2,8]
high_win = [60,120]
mm = PAC(data, low_win, high_win, fs=81920, visual=True)
print(mm)

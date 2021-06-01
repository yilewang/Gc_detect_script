#!/bin/usr/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
from scipy import signal
from findpeaks import PeakFinder

"""
This is a python file designed to investigate the relationship between G and oscillations.
@Author: Yile Wang
@Date: 05/28/2021
@Email: yile.wang@utdallas.edu

"""
# brain regions labels
regions = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R']


def fir_bandpass(data, fs, cut_off_low, cut_off_high, width=2.0, ripple_db=10.0):
    """
    The FIR bandpass filter
    """
    nyq_rate = fs / 2.0
    wid = width/nyq_rate
    N, beta = signal.kaiserord(ripple_db, wid)
    taps = signal.firwin(N, cutoff = [cut_off_low, cut_off_high],
                  window = 'hamming', pass_zero = False, fs=fs)
    filtered_signal = signal.lfilter(taps, 1.0, data)
    return filtered_signal, N



if __name__ == '__main__':
    # the sampling rate
    fs = 81920
    # the time
    samplinginterval = 1/fs
    t = np.arange(0, 1, samplinginterval)

    # the four study groups, Alzheimer's Disease, Mild Cognitive Impairment, Normal Control, Super Normal Control
    grp_pools = ['NC', 'MCI','AD','SNC']
    start = time.time()
    pdList = []
    for grp in grp_pools:

        # obtain the data path
        pth = 'C:/Users/Wayne/output/'+grp
        case_pools = os.listdir(pth)

        # iterate the case id.
        for caseid in case_pools:
            gRange = np.round(np.arange(0.001, 0.08, 0.001), 3)
            Gc = []
            Gm = []
            gamma_peaks = np.array([])
            theta_peaks = np.array([])
            for gm in gRange:
                dataFile = 'C:/Users/Wayne/tvb/LFP/'+grp+'/'+caseid+'/'+caseid+'_'+str(gm)+'.csv'
                
                # pandas read the data
                df = pd.read_csv(dataFile, index_col=0)

                # PCG left and right, channel 4-L, channel 5-R
                dfPCG = df.iloc[:, 4:6]

                # Other limbic channel, Rest
                df1 = df.iloc[:, 0:4]
                df2 = df.iloc[:, 6:16]
                dfRest = pd.concat([df1, df2], axis=1)

                # Gamma Band
                pcgGammaL, N = fir_bandpass(np.asarray(df['pCNG-L']), fs, 25.0, 100.0)
                pcgGammaR , N = fir_bandpass(np.asarray(df['pCNG-R']), fs, 25.0, 100.0)


                # Theta Band
                pcgThetaL, N= fir_bandpass(np.asarray(df['pCNG-L']), fs, 4.0, 8.0)
                pcgThetaR, N= fir_bandpass(np.asarray(df['pCNG-R']), fs, 4.0, 8.0)

                # delay
                delay = 0.5 * (N-1) / fs

                # peaks detection

                # rest of the regions
                avgRest = np.average(dfRest, axis = 1)
                peaksRest, _ = signal.find_peaks(avgRest[1000:], prominence=0.1) # to determine the Gc
                
                # Gamma signals
                peaksGammaR, _ = signal.find_peaks(df["pCNG-R"], height=np.max(pcgThetaR), prominence = 0.1)
                peaksGammaL, _ = signal.find_peaks(df["pCNG-L"], height=np.max(pcgThetaL), prominence = 0.1)
                
                # Theta signals
                func1 = PeakFinder(pcgThetaR)
                ThetaR = func1.findPeaks()
                #func1.peaks_plots(ThetaR)
                func2 = PeakFinder(pcgThetaL)
                ThetaL = func2.findPeaks()
                peaksThetaR = ThetaR[1::2]
                peaksThetaL = ThetaL[1::2]
                valleysThetaR = ThetaR[0::2]
                valleysThetaL = ThetaL[1::2]

                
                #visualization
                # fig, (ax1, ax2) = plt.subplots(2, figsize=(9,8))
                # fig.suptitle(caseid + "_filtered data_"+ str(gm))
                # ax1.plot(t, df['pCNG-R'], label = "Raw")                
                # ax1.plot(t[N-1:]-delay, pcgThetaR[N-1:], label = "Theta")
                # ax1.plot(t[N-1:]-delay, pcgGammaR[N-1:], label = "Gamma")
                # ax1.legend()
                # ax1.title.set_text('Theta&Gamma Fliter Signals')
                # ax2.plot(t[N-1:] - delay, pcgThetaR[N-1:], label = "Theta")
                # ax2.plot(ThetaR/fs - delay, pcgThetaR[ThetaR],'xg', label = "Peaks and Valleys")
                # ax2.title.set_text('Theta Frequency')
                # ax2.legend()
                # plt.show()

                # count the peaks
                gamma_peaks = np.append(gamma_peaks, len(peaksGammaR) + len(peaksGammaL))
                theta_peaks = np.append(theta_peaks, len(peaksThetaR)+ len(peaksThetaL))
                
                while len(peaksGammaR)+len(peaksGammaL) > 0 and len(peaksRest) == 0:
                    Gc.append(gm)
                    break
                while len(ThetaR) <=2 and len(ThetaL) <= 2:
                    Gm.append(gm)
                    break


            plt.figure(figsize=(9, 5))
            if not Gc:
                Gc = [0]
                Gc_label = "G critical = None"
            else: 
                Gc_label = "G critical = " + str(Gc[0])
            if not Gm:
                Gm = [0.079]
                Gm_label = "G max = " + str(Gm[0])
            else:
                Gm_label = "G max = " + str(Gm[0])
            plt.axvline(x=Gc[0], color='r', linestyle='dotted', label = Gc_label)
            plt.axvline(x=Gm[0], color='r', linestyle = 'dashed', label = Gm_label)            
            plt.plot(gRange, gamma_peaks, '*:g', label = "Gamma Frequency")
            plt.plot(gRange, theta_peaks, 'x:b', label = "Theta Frequency")
            plt.xticks(np.arange(0.001, 0.08, 0.005))
            plt.title(caseid+"_G ~ Gamma&Theta oscillation trend")
            plt.legend()

            # save pics
            save_path = grp+"_"+caseid+".png"
            plt.savefig(save_path)

            end = time.time()
            logging.warning('Duration: {}'.format(end - start))
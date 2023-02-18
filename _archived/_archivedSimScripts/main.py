import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtp
import os
import time
import logging
from scipy import signal
from findpeaks import PeakFinder
import scipy
"""
This script is designed as a tool for discovering Gc and Gmax.
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



def find_G(grp, caseid):
    # the sampling rate
    fs = 81920
    # the time
    samplinginterval = 1/fs
    t = np.arange(0, fs, 1)
    start = time.time()
    gRange = np.round(np.arange(0.001, 0.079, 0.001), 3)
    Gc = np.array([])
    Gmax = np.array([])
    for gm in gRange:
        # store the filename and prepare for reading the data
        dataFile = 'C:/Users/Wayne/tvb/LFP/'+grp+'/'+caseid+'/'+caseid+'_'+str(gm)+'.csv'
        # pandas read the data
        df = pd.read_csv(dataFile, index_col=0)
        dfL = df.iloc[:, 4]
        dfR = df.iloc[:, 5]

        # restareas
        df1 = df.iloc[:, 0:4]
        df2 = df.iloc[:, 6:16]
        dfRest = pd.concat([df1, df2], axis=1)

        # Gamma Band
        pcgGammaL, N = fir_bandpass(np.asarray(df['pCNG-L']), fs, 25.0, 100.0)
        pcgGammaR , N = fir_bandpass(np.asarray(df['pCNG-R']), fs, 25.0, 100.0)


        # Theta Band
        pcgThetaL, N= fir_bandpass(np.asarray(df['pCNG-L']), fs, 4.0, 8.0)
        pcgThetaR, N= fir_bandpass(np.asarray(df['pCNG-R']), fs, 4.0, 8.0)

        # Gamma signals
        GammaR, _ = signal.find_peaks(df["pCNG-R"], height=np.max(pcgThetaR), prominence = 0.1)
        GammaL, _ = signal.find_peaks(df["pCNG-L"], height=np.max(pcgThetaL), prominence = 0.1)
        
        tmpGR = [x for x in GammaR if x > 0.1*fs]
        tmpGL = [x for x in GammaL if x > 0.1*fs]
        peaksNum_gamma = len(tmpGR) + len(tmpGL)
        
        # Theta signals
        
        caseR = PeakFinder(pcgThetaR)
        ThetaR= caseR.findPeaks()
        caseL = PeakFinder(pcgThetaL)
        ThetaL = caseL.findPeaks()

        peaksNum_theta = len(ThetaR) + len(ThetaL)



        # Rest Areas
        avgRest = np.average(dfRest, axis = 1)
        peaksRest, _ = signal.find_peaks(avgRest[1000:], prominence=0.1) # to determine the Gc
        print(peaksNum_gamma, peaksNum_theta, peaksRest)
        if peaksNum_gamma >= 0 and len(peaksRest) < 1:
            Gc = np.append(Gc, gm)
        
        if peaksNum_theta < 1:
            Gmax = np.append(Gmax, gm)
    return Gc[0], Gmax[0]
        




if __name__ == '__main__':
    Gc, Gmax = find_G('AD', '4542A')
    print(Gc, Gmax)
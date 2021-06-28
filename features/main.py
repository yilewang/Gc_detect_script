import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
import time
import os
import logging

"""
@ Author: Yile Wang
This script is designed to capture all the lateralization features required for paper writing.
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

def valleyfinder(data, height=-1):
    """
    It is a new self-designed python function to detect peaks and valleys points
    in 1D local field potentials data, especially for slow waves(Theta band)

    """
    Num = len(data)
    Range = np.arange(0, Num, 1)
    ini = data[0]

    # list for peak and valley
    Gp = [] # for peak point
    Gv = [] # for valley point

    # to get all the local maxima and minima
    for i in Range[:-1]:
        # peaks
        while data[i] > ini:
            if data[i] > data[i+1]:
                Gp.append(i)
            ini = data[i]

        # valleys
        while data[i] < ini:
            if data[i] <= data[i+1]:
                Gv.append(i)
            ini = data[i]
    tmp = data[Gv]
    tmp_indx = np.argwhere(tmp < -1).ravel()
    Gv = np.array(Gv)
    Gvalleys = Gv[tmp_indx]
    return Gvalleys
    

def AmpCal(data, valleys, peaks):
    # calculate the amplitude from 0 to the first valleys
    amp_list = np.array([])
    len_valleys = len(valleys)
    v_range = np.arange(0, len(valleys)-1, 1)
    for ran in v_range:
        epo_range = [pe for pe in peaks if pe>valleys[ran] and pe<valleys[ran+1]]
        if len(epo_range) > 0:
            amp_range = np.max(data[epo_range])
            amp_list = np.append(amp_list, amp_range)
    return amp_list







if __name__ == '__main__':
    # read Gc and Go file
    coData = pd.read_excel('C:/Users/Wayne/tvb/TVB_workflow/new_g_oscillation/Gc_Go.xlsx', index_col=0)
    # the sampling rate
    fs = 81920
    # the time
    samplinginterval = 1/fs
    t = np.arange(0, fs, 1)
    tt = np.arange(0, 1, samplinginterval)
    # the four study groups, Alzheimer's Disease, Mild Cognitive Impairment, Normal Control, Super Normal Control
    grp_pools = ['AD','SNC','MCI', 'NC']
    start = time.time()
    pdList = []
    # fig, axs = plt.subplots(2, sharex = True, sharey = True, figsize=(12,8))
    # fig.suptitle("G frequency and Gamma")
    ax = 0
    col = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
    xx = 0
    freq = pd.DataFrame(columns=['grp','caseid','freqL', 'freqR'])
    for grp in grp_pools:
        # obtain the data path
        pth = 'C:/Users/Wayne/tvb/LFP/'+grp
        case_pools = os.listdir(pth)
        # iterate the case id.
        color = col[xx]
        for caseid in case_pools:
            try:
                # change it to Gc or Go
                gm = np.round(coData.loc[caseid, "Go"], 3)
                # store the filename and prepare for reading the data
                dataFile = 'C:/Users/Wayne/tvb/LFP/'+grp+'/'+caseid+'/'+caseid+'_'+str(gm)+'.csv'
                # pandas read the data
                df = pd.read_csv(dataFile, index_col=0)
                dfL = df.iloc[:, 4]
                dfR = df.iloc[:, 5]

                # Gamma Band
                pcgGammaL, N = fir_bandpass(np.asarray(df['pCNG-L']), fs, 35.0, 100.0)
                pcgGammaR , N = fir_bandpass(np.asarray(df['pCNG-R']), fs, 35.0, 100.0)

                # diff
                diffRL = np.array(pcgGammaR - pcgGammaL)


                # Theta Band
                pcgThetaL, N= fir_bandpass(np.asarray(df['pCNG-L']), fs, 4.0, 8.0)
                pcgThetaR, N= fir_bandpass(np.asarray(df['pCNG-R']), fs, 4.0, 8.0)

                # delay
                delay = 0.5 * (N-1) / fs

                # hilbert transform
                hil1 = hilbert(np.array(pcgGammaL))
                hil2 = hilbert(np.array(pcgGammaR))

                # envelope
                env1 = np.abs(hil1)
                env2 = np.abs(hil2)

                # Gamma peaks
                GammaR, _ = signal.find_peaks(df["pCNG-R"], height=np.max(pcgThetaR), prominence = 0.4)
                GammaL, _ = signal.find_peaks(df["pCNG-L"], height=np.max(pcgThetaL), prominence = 0.4)

                # Theta peaks
                ThetaR = valleyfinder(pcgThetaR, height=-1)
                ThetaL = valleyfinder(pcgThetaL, height=-1)

                ### Frequencies ###
                freq = freq.append({'grp':grp, 'caseid': caseid, 'freqL': len(GammaL), 'freqR':len(GammaR)}, ignore_index=True)

            except FileNotFoundError:
                continue
            except KeyError:
                continue
    freq.to_csv('freq_Go.csv')



                # visualization
                # fig, (ax1, ax2) = plt.subplots(2, figsize=(15,10))
                # fig.suptitle(grp+'_'+caseid + "_filtered data_"+ str(gm))
                # ax1.plot(tt, df['pCNG-R'], label = "Raw")
                # ax1.plot(tt[N-1:]-delay, pcgThetaR[N-1:], label = "theta")
                # #ax1.plot(t, env2, label = 'Envelope')             
                # #ax1.plot(t, pcgGammaR, label = "Gamma")
                # ax1.plot(GammaR[GammaR > N-1]/fs, df['pCNG-R'][GammaR[GammaR > N-1]], 'x:r')
                # ax1.plot(ThetaR[ThetaR > N-1]/fs-delay, pcgThetaR[ThetaR[ThetaR > N-1]], 'x:g')
                # ax1.legend()
                # ax1.title.set_text('Theta&Gamma Fliter Signals_R')
                # ax2.plot(tt, df['pCNG-L'], label = "Raw")
                # ax2.plot(tt[N-1:]-delay, pcgThetaL[N-1:], label = "theta")
                # #ax1.plot(t, env2, label = 'Envelope')             
                # #ax1.plot(t, pcgGammaR, label = "Gamma")
                # ax2.plot(GammaL[GammaL > N-1]/fs, df['pCNG-L'][GammaL[GammaL > N-1]], 'x:r')
                # ax2.plot(ThetaL[ThetaL > N-1]/fs-delay, pcgThetaL[ThetaL[ThetaL > N-1]], 'x:g')
                # ax2.legend()
                # ax2.title.set_text('Theta&Gamma Fliter Signals_L')
                # #plt.show()
                # pt = grp + '_' + caseid + '_' + str(gm) +'.png'
                # #plt.savefig(pt)


            # ### Amplitude: The distance from the center of motion to either extreme ###
            # ThetaL = np.append(ThetaL, fs)
            # ThetaR = np.append(ThetaR, fs)
            # amp_r = AmpCal(df['pCNG-R'], ThetaR, GammaR)
            # amp_l = AmpCal(df['pCNG-L'], ThetaL, GammaL)
            # ampL = np.mean(amp_l)
            # ampR = np.mean(amp_r)
            # amp = amp.append({'grp':grp, 'caseid': caseid, 'ampL': ampL, 'ampR':ampR}, ignore_index=True)











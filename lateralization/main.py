import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
import scipy.stats as stats
import time
import os
import logging
from statsmodels.tsa.stattools import grangercausalitytests
"""
This script is designed to do all the lateralization analysis between Alzheimer's Disease and normal groups
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
    # read Gc and Go file
    coData = pd.read_excel('C:/Users/Wayne/tvb/TVB_workflow/new_g_oscillation/Gc_Go.xlsx', index_col=0)
    # the sampling rate
    fs = 81920
    # the time
    samplinginterval = 1/fs
    t = np.arange(0, fs, 1)
    # the four study groups, Alzheimer's Disease, Mild Cognitive Impairment, Normal Control, Super Normal Control
    grp_pools = ['SNC','NC','MCI', 'AD']
    start = time.time()
    pdList = []
    # fig, axs = plt.subplots(2, sharex = True, sharey = True, figsize=(12,8))
    # fig.suptitle("G frequency and Gamma")
    ax =0
    col = ['b', 'r']
    xx = 0
    for grp in grp_pools:
        # obtain the data path
        pth = 'C:/Users/Wayne/tvb/LFP/'+grp
        case_pools = os.listdir(pth)
        # iterate the case id.
        color = col[xx]
        for caseid in case_pools:
            gRange = np.round(np.arange(coData.loc[caseid, "Gc"], coData.loc[caseid, "Gmax"], 0.001), 3)
            for gm in gRange:
                # store the filename and prepare for reading the data
                dataFile = 'C:/Users/Wayne/tvb/LFP/'+grp+'/'+caseid+'/'+caseid+'_'+str(gm)+'.csv'
                # pandas read the data
                df = pd.read_csv(dataFile, index_col=0)
                dfL = df.iloc[:, 4]
                dfR = df.iloc[:, 5]

                # Gamma Band
                pcgGammaL, N = fir_bandpass(np.asarray(df['pCNG-L']), fs, 25.0, 100.0)
                pcgGammaR , N = fir_bandpass(np.asarray(df['pCNG-R']), fs, 25.0, 100.0)


                # Theta Band
                pcgThetaL, N= fir_bandpass(np.asarray(df['pCNG-L']), fs, 4.0, 8.0)
                pcgThetaR, N= fir_bandpass(np.asarray(df['pCNG-R']), fs, 4.0, 8.0)

                #  # delay
                # delay = 0.5 * (N-1) / fs

                #hilbert transform
                al1 = np.angle(hilbert(dfL),deg=False)
                al2 = np.angle(hilbert(dfR),deg=False)
                phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)

                # Plot results
                f,ax = plt.subplots(3,1,sharex=True)
                ax[0].plot(dfL,color='r',label='pCNG-L')
                ax[0].plot(dfR,color='b',label='pCNG-R')
                ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)
                info = grp + '_'+ caseid + '_' + str(gm)
                ax[0].set( title=info)
                ax[1].plot(al1,color='r')
                ax[1].plot(al2,color='b')
                ax[1].set(ylabel='Angle',title='Angle at each Timepoint')
                ax[2].plot(phase_synchrony)
                ax[2].set(title='Instantaneous Phase Synchrony',xlabel='Time',ylabel='Phase Synchrony')
                plt.tight_layout()
                plt.show()

                # coherence analysis
                # f, Cxy = signal.coherence(np.asarray(pcgGammaL), np.asarray(pcgGammaR), 200, nperseg=1024)
                # axs[ax].semilogy(f, Cxy, color = color)
                # axs[ax].set_xlabel('Frequency [Hz]')
                # axs[ax].set_ylabel('Coherence')
                # axs[ax].set_title('Coherence Analysis')

                # cross correlation
                # LL = pcgGammaL
                # RR = pcgGammaR
                # # fig, axs = plt.subplots(1, figsize=(10,5))
                # xcorr = signal.correlate(np.asarray(LL), np.asarray(RR), mode = 'same') / np.sqrt(signal.correlate(np.asarray(LL), np.asarray(LL), mode='same')[int(fs/2)] * signal.correlate(np.asarray(RR), np.asarray(RR), mode='same')[int(fs/2)])
                # lags = np.linspace(-0.5*fs, 0.5*fs, fs)
                # axs[ax].plot(lags, xcorr, color=color)
                # axs.set_title("Cross Correlation")
                # axs.set_xlabel('Lags')
                # axs.set_ylabel('Correlation Coeficient')
                # plt.show()

                # Granger Causality
                #grangercausalitytests(df[['pCNG-L', 'pCNG-R']], 100, addconst=True, verbose=True)
                
        end = time.time()
        logging.warning('Duration: {}'.format(end - start))

        #ax+=1
        xx+=1

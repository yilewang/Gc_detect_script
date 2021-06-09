#!/bin/usr/python

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


def main(t, y1, ax, color): #y2):
    # Non-linear Fit
    A, K, C = fit_exp_nonlinear(t, y1)
    fit_y = model_func(t, A, K, C)
    plot(ax, t, y1, fit_y, (A, K, C), color)
    title = "Non-linear Fit_Gamma_" + grp
    ax.set_title(title)

    # A1, K1, C1 = fit_exp_nonlinear(t, y2)
    # fit_y1 = model_func(t, A1, K1, C1)
    # plot(ax2, t, y2, fit_y1, (A1, K1, C1), color)
    # title1 = "Non-linear Fit_Theta"
    # ax2.set_title(title1)

    # # Linear Fit (Note that we have to provide the y-offset ("C") value!!
    # A, K = fit_exp_linear(t, y, C)
    # fit_y = model_func(t, A, K, C)
    # plot(ax1, t, y, fit_y, (A, K, C))
    # ax1.set_title('Linear Fit')

    return A
    

def model_func(t, A, K, C):
    return A * np.exp(K * t) + C

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = scipy.optimize.curve_fit(model_func, t, y, maxfev=5000)
    A, K, C = opt_parms
    return A, K, C


# def fit_exp_linear(t, y, C):
#     y = y - C
#     y = np.log(y)
#     K, A_log = np.polyfit(t, y, 1)
#     A = np.exp(A_log)
#     return A, K

def plot(ax, t, y, fit_y, fit_parms, color):
    A, K, C = fit_parms
    ax.plot(t, y, marker=".", markersize=5, color = color, alpha = 0.3)
    ax.plot(t, fit_y, color = color, alpha = 0.8)
    #ax.legend()
    #ax.plot(t, fit_y, 'b-',
    #  label='Fitted Function:\n $y = %0.2f e^{%0.2f t} + %0.2f$' % (A, K, C))
    #ax.legend(bbox_to_anchor=(1.05, 1.1), fancybox=True, shadow=True)



if __name__ == '__main__':
    # read Gc and Go file
    coData = pd.read_excel('C:/Users/Wayne/tvb/TVB_workflow/new_g_oscillation/Gc_Go.xlsx', index_col=0)
    # the sampling rate
    fs = 81920
    # the time
    samplinginterval = 1/fs
    t = np.arange(0, 1, samplinginterval)
    # the four study groups, Alzheimer's Disease, Mild Cognitive Impairment, Normal Control, Super Normal Control
    grp_pools = ['SNC','NC', 'MCI', 'AD']
    start = time.time()
    pdList = []
    fig, axs = plt.subplots(4, sharex = True, sharey = True, figsize=(12,8))
    fig.suptitle("G frequency and Gamma")
    ax=0
    slope = pd.DataFrame(columns = ['Grp', 'caseid','A','K','C', 'Gmax'])
    Go = pd.DataFrame(columns = ['Grp', 'caseid', 'Go', 'L_freq', 'R_freq'])
    for grp in grp_pools:
        # obtain the data path
        pth = 'C:/Users/Wayne/tvb/LFP/'+grp
        case_pools = os.listdir(pth)
        mtpColors = list(mtp.colors.cnames.values())
        # iterate the case id.
        count=10
        for caseid in case_pools:
            gRange = np.round(np.arange(coData.loc[caseid, 'Gc'], 0.079, 0.001), 3)
            gamma_peaks = np.array([])
            theta_peaks = np.array([])
            Gm = np.array([])
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
                # # rest of the regions
                # avgRest = np.average(dfRest, axis = 1)
                # peaksRest, _ = signal.find_peaks(avgRest[1000:], prominence=0.1) # to determine the Gc
                
                # Gamma signals
                GammaR, _ = signal.find_peaks(df["pCNG-R"], height=np.max(pcgThetaR), prominence = 0.1)
                GammaL, _ = signal.find_peaks(df["pCNG-L"], height=np.max(pcgThetaL), prominence = 0.1)
                
                tmpGR = [x for x in GammaR if x > 0.1*fs]
                tmpGL = [x for x in GammaL if x > 0.1*fs]
                peaksNum_gamma = len(tmpGR) + len(tmpGL)
                gamma_peaks = np.append(gamma_peaks, peaksNum_gamma)
                
                # Theta signals
                
                caseR = PeakFinder(pcgThetaR)
                ThetaR= caseR.findPeaks()
                caseL = PeakFinder(pcgThetaL)
                ThetaL = caseL.findPeaks()

                peaksNum_theta = len(ThetaR) + len(ThetaL)
                theta_peaks = np.append(theta_peaks, peaksNum_theta)

                if gm == coData.loc[caseid, 'Go']:
                    Go2 = pd.DataFrame([[grp, caseid, gm, len(GammaR), len(GammaL)]], columns = ['Grp', 'caseid', 'Go', 'L_freq', 'R_freq'])
                    Go = Go.append(Go2, ignore_index = True)
                    

                
                #visualization
                # fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(9,8))
                # fig.suptitle(caseid + "_filtered data_"+ str(gm))
                # ax1.plot(t, df['pCNG-R'], label = "Raw")                
                # ax1.plot(t[N-1:]-delay, pcgThetaR[N-1:], label = "Theta")
                # ax1.plot(t[N-1:]-delay, pcgGammaR[N-1:], label = "Gamma")
                # ax1.plot(GammaR/fs, df['pCNG-R'][GammaR], 'x:r')
                # ax1.legend()
                # ax1.title.set_text('Theta&Gamma Fliter Signals')
                # ax2.plot(t[N-1:], pcgThetaL[N-1:], label = "Theta")
                # if len(ThetaL) > 0: 
                #     ax2.plot(ThetaL/fs, pcgThetaL[ThetaL],'xg', label = "Peaks and Valleys")
                # ax2.title.set_text('Theta Frequency')
                # ax2.legend()
                # ax3.plot(t[N-1:], pcgThetaR[N-1:], label = "Theta")
                # if len(ThetaR) > 0:
                #     ax3.plot(ThetaR/fs, pcgThetaR[ThetaR],'xg', label = "Peaks and Valleys")
                # ax3.title.set_text('Theta Frequency')
                # ax3.legend()
                # plt.show()
                while peaksNum_theta == 0:
                    Gm = np.append(Gm, gm)
                    break
            if len(Gm) < 1:
                Gm = [0.079]
            cmRange = np.arange(coData.loc[caseid, 'Gc'], Gm[0], 0.001)
            try:
                #main(cmRange, gamma_peaks[0:len(cmRange)], axs[ax], mtpColors[count])
                A, K, C = fit_exp_nonlinear(cmRange, gamma_peaks[0:len(cmRange)])
            except:
                continue
            count+=1
            slope2 = pd.DataFrame([[grp, caseid, A, K, C, Gm[0]]], columns = ['Grp', 'caseid','A','K','C', 'Gmax'])
            slope = slope.append(slope2, ignore_index=True)



            # plt.figure(figsize=(9, 5))
            # plt.plot(cmRange, gamma_peaks[0:len(cmRange)], '*:g', label = "Gamma Frequency")
            # plt.plot(cmRange, theta_peaks[0:len(cmRange)], 'x:r', label = "Theta Frequency")
            # Go = 'G optimal = ' + str(coData.loc[caseid, 'Go'])
            # plt.axvline(coData.loc[caseid, 'Go'], linestyle = "dotted",color = "b",label = Go)
            # plt.xticks(cmRange)
            # plt.title(caseid+"_G ~ Gamma oscillation trend")
            # plt.legend()
            # plt.show()

            # # save pics
            # save_path = grp+"_"+caseid+".png"
            # plt.savefig(save_path)

            end = time.time()
            logging.warning('Duration: {}'.format(end - start))
        ax+=1
    Go.to_csv(r'C:/Users/Wayne/tvb/Go_freq.csv', index=False, header = True)
    # save pics
    # save_path = "Gamma_G.png"
    # plt.savefig(save_path) 
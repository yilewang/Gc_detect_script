import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
import time
import os
import logging
import sys
sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\functions')
# from permutation import PermutationTest
# from bootstrap import BootstrapTest

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


def peaks_valleys_finder(data):
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
    return Gp, Gv

def valleysfinder(data, Gv, height=-1):
    """
    try to exclude some valley points lower than threshold
    """
    tmp = data[Gv]
    tmp_indx = np.argwhere(tmp < height).ravel()
    Gv = np.array(Gv)
    Gvalleys = Gv[tmp_indx]
    return Gvalleys
    

def AmpAbs(data, filter_data, valleys, Gamma_peaks, Theta_peaks, delay_fs):
    """
    calculate the absolute amplitude 
    """
    # calculate the amplitude from 0 to the first valleys
    amp_list = np.array([])
    len_valleys = len(valleys)
    v_range = np.arange(0, len(valleys)-1, 1)
    for ran in v_range:
        tmp_pe = Gamma_peaks[(valleys[ran] < Gamma_peaks) & (Gamma_peaks < valleys[ran+1])]
        if len(tmp_pe) > 0:
            valleys_value = np.mean(filter_data[valleys - int(delay_fs)])
            pm = np.max(data[tmp_pe]-valleys_value) # data - mean of the valleys
            amp_list = np.append(amp_list, pm)
        else:
            tmp_tpe = Theta_peaks[(valleys[ran] < Theta_peaks) & (Theta_peaks < valleys[ran+1])]
            if len(tmp_tpe) >0:
                valleys_value = np.mean(filter_data[valleys - int(delay_fs)])
                tmp_tpeak_cyc = data[tmp_tpe] - valleys_value
                amp_list = np.append(amp_list, tmp_tpeak_cyc)
            else:
                amp_list = np.append(amp_list, 0)
    return amp_list


def AmpCombine(data, filter_data, Gamma_peaks, Theta_peaks, valleys):
    """
    calculate the value data-filter_data.
    """
    amp_list = np.array([])
    v_range = np.arange(0, len(valleys)-1, 1)
    for ran in v_range:
        tmp_pe = Gamma_peaks[(valleys[ran] < Gamma_peaks) & (Gamma_peaks < valleys[ran+1])]
        if len(tmp_pe) > 0:
            tmp_peak_cyc = np.array([])
            for single in tmp_pe:
                single_amp = data[single] - filter_data[single-int(delay_fs)] # 
                tmp_peak_cyc = np.append(tmp_peak_cyc, single_amp)
            amp_list = np.append(amp_list,np.mean(tmp_peak_cyc))
        else:
            tmp_tpe = Theta_peaks[(valleys[ran] < Theta_peaks) & (Theta_peaks < valleys[ran+1])]
            if len(tmp_tpe) >0:
                tmp_tpeak_cyc = data[tmp_tpe] - filter_data[tmp_tpe-int(delay_fs)]
                amp_list = np.append(amp_list, tmp_tpeak_cyc)
            else:
                amp_list = np.append(amp_list, 0)
    return amp_list


def AmpPro(data, filter_data, Gamma_peaks, Theta_peaks, valleys):
    """
    calculate the proportional amplitude here
    """
    amp_list_gamma = np.array([])
    amp_list_theta = np.array([])
    v_range = np.arange(0, len(valleys)-1, 1)
    for ran in v_range:
        tmp_pe = Gamma_peaks[(valleys[ran] < Gamma_peaks) & (Gamma_peaks < valleys[ran+1])]
        if len(tmp_pe) > 0:
            tmp_cyc_gamma = np.array([])
            tmp_cyc_theta = np.array([])
            for single in tmp_pe:
                overall_amp = data[single] - np.mean(filter_data[valleys])
                single_gamma = data[single] - filter_data[single]
                single_theta = filter_data[single] - np.mean(filter_data[valleys])
                proportional_gamma = single_gamma / overall_amp
                proportional_theta = single_theta / overall_amp
                tmp_cyc_gamma = np.append(tmp_cyc_gamma, proportional_gamma)
                tmp_cyc_theta = np.append(tmp_cyc_theta, proportional_theta)
            amp_list_gamma = np.append(amp_list_gamma, tmp_cyc_gamma.mean())
            amp_list_theta = np.append(amp_list_theta, tmp_cyc_theta.mean())
        else:
            tmp_tpe = Theta_peaks[(valleys[ran] < Theta_peaks) & (Theta_peaks < valleys[ran+1])]
            if len(tmp_tpe) >0:
                amp_list_gamma = np.append(amp_list_gamma, 0)
                amp_list_theta = np.append(amp_list_theta, 1)
            else:
                amp_list_gamma = np.append(amp_list_gamma, 0)
                amp_list_theta = np.append(amp_list_theta, 0)
    return amp_list_gamma, amp_list_theta

def DelayCal(filter_data_left, filter_data_right, valley_left, valley_right, fs):
    """
    calculate the delay
    """
    delay_list = np.array([])
    Theta_peak_left,_ = signal.find_peaks(filter_data_left, prominence=0.1, height= -1.25)
    Theta_peak_right,_ = signal.find_peaks(filter_data_right, prominence=0.1, height=-1.25)
    v_range_left = np.arange(0, len(valley_left)-1, 1)
    v_range_right = np.arange(0, len(valley_right)-1, 1)

    # left peaks 
    left_peaks_list = np.array([])
    for ran in v_range_left:
        tmp_peaks_left = Theta_peak_left[(valley_left[ran] < Theta_peak_left) & (Theta_peak_left < valley_left[ran+1])]
        if tmp_peaks_left.shape[0] > 0:
            tmp_single_peak_left = tmp_peaks_left[0]
            left_peaks_list = np.append(left_peaks_list, tmp_single_peak_left)

    # right peaks
    right_peaks_list = np.array([])
    for ran in v_range_right:
        tmp_peaks_right = Theta_peak_right[(valley_right[ran] < Theta_peak_right) & (Theta_peak_right < valley_right[ran+1])]
        if tmp_peaks_right.shape[0] >0:
            tmp_single_peak_right = tmp_peaks_right[0]
            right_peaks_list = np.append(right_peaks_list, tmp_single_peak_right)
    
    # calculate delay
    if left_peaks_list.shape[0] > 0:
        if left_peaks_list.shape[0] > right_peaks_list.shape[0]:
            valid_range = right_peaks_list.shape[0]
            Delay = np.abs(right_peaks_list - left_peaks_list[0:valid_range])
        else:
            valid_range = left_peaks_list.shape[0]
            Delay = np.abs(right_peaks_list[0:valid_range] - left_peaks_list)
        return np.sum(Delay/fs)
    else:
        Delay = 3.26 + 1 + ((6 - right_peaks_list.shape[0])/ 3)
        return Delay


def Thetapeaksfinder(valleys, peaks):
    len_valleys = len(valleys)
    v_range = np.arange(0, len(valleys)-1, 1)
    freq_count = 0
    for ran in v_range:
        epo_range = peaks[(valleys[ran] < peaks) & (peaks < valleys[ran+1])]
        if len(epo_range) > 0:
            freq_count += 1
    return freq_count


############################################################################
# visualization functions

def visual_raw_filter():
    # visualization
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15,10))
    fig.suptitle(grp+'_'+caseid + "_filtered data_"+ str(gm))
    ax1.plot(tt, df['pCNG-R'], label = "Raw")
    ax1.plot(tt[N-1:]-delay, pcgThetaR[N-1:], label = "theta")
    #ax1.plot(t, env2, label = 'Envelope')             
    #ax1.plot(t, pcgGammaR, label = "Gamma")
    ax1.plot(GammaR[GammaR > N-1]/fs, df['pCNG-R'][GammaR[GammaR > N-1]], 'x:r')
    ax1.plot(valleysR[valleysR > N-1]/fs-delay, pcgThetaR[valleysR[valleysR > N-1]], 'x:g')
    ax1.legend()
    ax1.title.set_text('Theta&Gamma Fliter Signals_R')
    ax2.plot(tt, df['pCNG-L'], label = "Raw")
    ax2.plot(tt[N-1:]-delay, pcgThetaL[N-1:], label = "theta")
    #ax1.plot(t, env2, label = 'Envelope')             
    #ax1.plot(t, pcgGammaR, label = "Gamma")
    ax2.plot(GammaL[GammaL > N-1]/fs, df['pCNG-L'][GammaL[GammaL > N-1]], 'x:r')
    ax2.plot(valleysL[valleysL > N-1]/fs-delay, pcgThetaL[valleysL[valleysL > N-1]], 'x:g')
    ax2.legend()
    ax2.title.set_text('Theta&Gamma Fliter Signals_L')
    plt.show()
    # pt = grp + '_' + caseid + '_' + str(gm) +'.png'
    # #plt.savefig(pt)
    
def visual_violin():
    # original data of Freq
    key1 = 'ampL'
    sns.boxplot(x='grp', y=key1, data=amp,  order=[ "SNC", "NC", "MCI", "AD"], palette=col, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
    sns.stripplot(y=key1, 
                x="grp", 
                data=amp, color='black', order=[ "SNC", "NC", "MCI", "AD"], edgecolor='gray')
    plt.title(key1)
    plt.show()

def visual_bootstrap():

    # create bootstrap data set
    AD_sample = np.hstack(amp.loc[amp['grp'].isin(['AD']), [key1]].values)
    MCI_sample = np.hstack(amp.loc[amp['grp'].isin(['MCI']), [key1]].values)
    NC_sample = np.hstack(amp.loc[amp['grp'].isin(['NC']), [key1]].values)
    SNC_sample = np.hstack(amp.loc[amp['grp'].isin(['SNC']), [key1]].values)

    PermutationTest(SNC_sample, AD_sample, 5000, True)
    PermutationTest(NC_sample, AD_sample, 5000, True)
    PermutationTest(MCI_sample, AD_sample, 5000, True)

    ad_CI, ad_dis = BootstrapTest(AD_sample, 1000)
    mci_CI, mci_dis = BootstrapTest(MCI_sample, 1000)
    nc_CI, nc_dis = BootstrapTest(NC_sample, 1000)
    snc_CI, snc_dis = BootstrapTest(SNC_sample, 1000)

    plt.figure(figsize=(15, 5))
    plt.title('Bootstrap of four groups')

    plt.hist(snc_dis, bins='auto', color= "#66CDAA", label='SNC', alpha = 0.5,histtype='bar', ec='black')
    # plt.axvline(x=np.round(snc_CI[0],3), label='CI at {}'.format(np.round(snc_CI,3)),c="#FFA15A", linestyle = 'dashed')
    # plt.axvline(x=np.round(snc_CI[1],3),  c="#FFA15A", linestyle = 'dashed')
    plt.hist(nc_dis, bins='auto', color="#4682B4", label='NC', alpha = 0.5,histtype='bar', ec='black')
    # plt.axvline(x=np.round(nc_CI[0],3), label='CI at {}'.format(np.round(nc_CI,3)),c="#AB63FA", linestyle = 'dashed')
    # plt.axvline(x=np.round(nc_CI[1],3),  c="#AB63FA", linestyle = 'dashed')
    plt.hist(mci_dis, bins='auto', color="#AB63FA", label='MCI', alpha = 0.5,histtype='bar', ec='black')
    # plt.axvline(x=np.round(mci_CI[0],3), label='CI at {}'.format(np.round(mci_CI,3)),c="#4682B4", linestyle = 'dashed')
    # plt.axvline(x=np.round(mci_CI[1],3),  c="#4682B4", linestyle = 'dashed')
    plt.hist(ad_dis, bins='auto', color="#FFA15A", label='AD', alpha = 0.5,histtype='bar', ec='black')
    # plt.axvline(x=np.round(ad_CI[0],3), label='CI at {}'.format(np.round(ad_CI,3)),c="#66CDAA", linestyle = 'dashed')
    # plt.axvline(x=np.round(ad_CI[1],3),  c="#66CDAA", linestyle = 'dashed')
    plt.legend()
    plt.show()
    
    print(freq.loc[freq['grp'].isin(['MCI']), ['freqL_Gamma']])

# visualization functions end
#########################################################################



if __name__ == '__main__':
    # read Gc and Go file
    laptop = r"C:\Users\wayne\OneDrive - The University of Texas at Dallas\tvb\stat_data"
    desktop = 'C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx'
    coData = pd.read_excel(desktop, index_col=0)
    # the sampling rate
    fs = 81920

    # the time
    samplinginterval = 1/fs
    t = np.arange(0, fs, 1)
    tt = np.arange(0, 1, samplinginterval)

    # the four study groups, Alzheimer's Disease, Mild Cognitive Impairment, Normal Control, Super Normal Control
    grp_pools = ['SNC','NC','MCI', 'AD']
    start = time.time()
    pdList = []
    # fig, axs = plt.subplots(2, sharex = True, sharey = True, figsize=(12,8))
    # fig.suptitle("G frequency and Gamma")
    ax = 0
    col = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
    xx = 0
    freq = pd.DataFrame(columns=['grp','caseid','freqL_Gamma','freqR_Gamma', 'freqL_Theta', 'freqR_Theta'])
    amp = pd.DataFrame(columns=['grp','caseid','ampL','ampR'])
    amp_pro = pd.DataFrame(columns=['grp','caseid','ampL_gamma','ampR_gamma','ampL_theta','ampR_theta'])
    all_delay = pd.DataFrame(columns=['grp','caseid','delay'])
    mix = pd.DataFrame()
        #columns=['grp','caseid','freqL_Gamma','freqR_Gamma', 'freqL_Theta', 'freqR_Theta', 'ampL_abs','ampR_abs','ampL_combine','ampR_combine', 'delay'])
    for grp in grp_pools:
        # obtain the data path
        pth = 'C:/Users/Wayne/tvb/LFP/'+grp
        #case_pools = ['3168A']
        case_pools = os.listdir(pth)
        # iterate the case id.
        color = col[xx]
        for caseid in case_pools:
            try:
                # change it to Gc or Go
                gm = np.round(coData.loc[caseid, "Gc"], 3)
                # store the filename and prepare for reading the data
                dataFile = 'C:/Users/Wayne/tvb/LFP/'+grp+'/'+caseid+'/'+caseid+'_'+str(gm)+'.csv'
                # pandas read the data
                df = pd.read_csv(dataFile, index_col=0)
                dfL = df.iloc[:, 4]
                dfR = df.iloc[:, 5]
                # spectrogram representation
                plt.figure(figsize=(10,10))
                powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(dfR, Fs=fs)
                plt.xlabel('Time')
                plt.ylabel('Frequency')
                plt.show()
                break
                # Gamma Band
                pcgGammaL, N = fir_bandpass(np.asarray(df['pCNG-L']), fs, 35.0, 100.0)
                pcgGammaR , N = fir_bandpass(np.asarray(df['pCNG-R']), fs, 35.0, 100.0)
                # diff
                diffRL = np.array(pcgGammaR - pcgGammaL)
                # Theta Band
                pcgThetaL, N= fir_bandpass(np.asarray(df['pCNG-L']), fs, 1.0, 10.0)
                pcgThetaR, N= fir_bandpass(np.asarray(df['pCNG-R']), fs, 1.0, 10.0)
                # delay
                delay = 0.5 * (N-1) / fs
                delay_fs = 0.5 * (N-1)
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
                ThetaL, _ = signal.find_peaks(pcgThetaL, prominence=0.2)
                ThetaR, _ = signal.find_peaks(pcgThetaR, prominence=0.2)
                # Theta valleys
                _, vtmpR = peaks_valleys_finder(pcgThetaR)
                _, vtmpL = peaks_valleys_finder(pcgThetaL)
                valleysR = valleysfinder(pcgThetaR, vtmpR, height=-1)
                valleysL = valleysfinder(pcgThetaL, vtmpL, height=-1)

                ##########################################
                ## Amplitude: The distance from the center of motion to either extreme
                ##########################################
                # the valleys (add the end point into the valleys)
                valleysL = np.append(valleysL, fs-1)
                valleysR = np.append(valleysR, fs-1)

                ### the absolute value of the pCNG amplitude
                amp_r = AmpAbs(df['pCNG-R'],pcgThetaR, valleysR, GammaR, ThetaR, delay_fs)
                amp_l = AmpAbs(df['pCNG-L'], pcgThetaL, valleysL, GammaL, ThetaL, delay_fs)
                ampL_abs = np.mean(amp_l)
                ampR_abs = np.mean(amp_r)

                # # the combine value of the pCNG amplitude
                # amp_r = AmpCombine(df['pCNG-R'], pcgThetaR, GammaR, ThetaR, valleysR)
                # amp_l = AmpCombine(df['pCNG-L'], pcgThetaL, GammaL, ThetaL, valleysL)

                # # take the mean
                # ampL_combine = np.mean(amp_l)
                # ampR_combine = np.mean(amp_r)

                # to dataframe
                amp = amp.append({'grp':grp, 'caseid': caseid, 'ampL': ampL_abs, 'ampR':ampR_abs}, ignore_index=True)

                #### the proportional value of the pCNG
                # amp_r_gamma, amp_r_theta  = AmpPro(df['pCNG-R'], pcgThetaR, GammaR, ThetaR, valleysR)
                # amp_l_gamma, amp_l_theta  = AmpPro(df['pCNG-L'], pcgThetaL, GammaL, ThetaL, valleysL)

                # ampL_gamma = np.mean(amp_l_gamma)
                # ampL_theta = np.mean(amp_l_theta)
                # ampR_gamma = np.mean(amp_r_gamma)
                # ampR_theta = np.mean(amp_r_theta)
                # amp_pro = amp_pro.append({'grp':grp, 'caseid': caseid, 'ampL_gamma': ampL_gamma, 'ampR_gamma':ampR_gamma, 'ampL_theta':ampL_theta, 'ampR_theta':ampR_theta}, ignore_index=True)

                #################################################
                ### Frequencies
                ###################################################
                GammaL_num = len(GammaL)
                GammaR_num = len(GammaR)
                if GammaL_num >= 5:
                    ThetaL_num = Thetapeaksfinder(valleysL, GammaL)
                else:
                    tt_tmp_l, _ = signal.find_peaks(pcgThetaL, prominence = 0.2)
                    ThetaL_num = len(tt_tmp_l)
                
                if GammaR_num >= 5:
                    ThetaR_num = Thetapeaksfinder(valleysR, GammaR)
                else:
                    tt_tmp_r, _ =  signal.find_peaks(pcgThetaR, prominence = 0.2)
                    ThetaR_num  = len(tt_tmp_r)

                freq = freq.append({'grp':grp, 'caseid': caseid, 'freqL_Gamma':GammaL_num,'freqR_Gamma':GammaR_num, 'freqL_Theta':ThetaL_num, 'freqR_Theta':ThetaR_num}, ignore_index=True)

                ######################################################
                ############ delay
                ######################################################
                subj_delay = DelayCal(pcgThetaL, pcgThetaR, valleysL, valleysR, fs)
                all_delay = all_delay.append({'grp':grp, 'caseid': caseid, 'delay':subj_delay}, ignore_index=True)

                #######################################################
                ############LI
                ######################################################
                # freqL_Gamma=min(GammaL_num, GammaR_num)
                # freqR_Gamma=max(GammaR_num, GammaL_num) 
                # freqL_Theta= min(ThetaL_num, ThetaR_num)
                # freqR_Theta = max(ThetaR_num, ThetaL_num) 
                # ampLside=min(ampL_abs, ampR_abs)
                # ampRside=max(ampR_abs, ampL_abs)

                freqL_Gamma=GammaL_num
                freqR_Gamma=GammaR_num 
                freqL_Theta= ThetaL_num
                freqR_Theta = ThetaR_num
                ampLside=ampL_abs
                ampRside=ampR_abs


                try:
                    LI_freq_gamma = (freqR_Gamma - freqL_Gamma)/(freqR_Gamma+freqL_Gamma)
                    LI_freq_theta= (freqR_Theta - freqL_Theta)/(freqL_Theta+freqR_Theta)
                    LI_amp= (ampRside - ampLside) / (ampRside + ampLside) 
                    LI_mix_freq = ((freqR_Gamma/freqR_Theta) - (freqL_Gamma/freqL_Theta))/((freqR_Gamma/freqR_Theta) + (freqL_Gamma/freqL_Theta))
                except ZeroDivisionError:
                    LI_mix_freq = 1


                ### mix table
                mix = mix.append({'grp':grp, 
                'caseid': caseid, 
                'freqL_Gamma':freqL_Gamma,
                'freqR_Gamma':freqR_Gamma, 
                'freqL_Theta':freqL_Theta, 
                'freqR_Theta':freqR_Theta, 
                'ampL_abs':ampLside,
                'ampR_abs':ampRside,
                # 'ampL_pro_gamma':amp_l_gamma,
                # 'ampR_pro_gamma':amp_r_gamma,
                # 'ampL_pro_theta':amp_l_theta, 
                # 'ampR_pro_theta':amp_r_theta,
                'LI_freq_gamma':LI_freq_gamma,
                'LI_freq_theta': LI_freq_theta,
                'LI_amp': LI_amp,
                'LI_mix_freq': LI_mix_freq,
                'delay':subj_delay}, 
                ignore_index=True, sort=False)

            except FileNotFoundError:
                continue
            except KeyError:
                continue
    # amp.to_csv('amp_abs.csv')
    # # amp.to_csv('amp_combine.csv')
    # amp_pro.to_csv('amp_pro_final.csv')
    # freq.to_excel('freq.xlsx')
    mix.to_excel('mix.xlsx')

    

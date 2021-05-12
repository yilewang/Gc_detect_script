# Inter-hemisphere dynamic: Python script
# Author: Yile Wang
# Date: 10.26.2020
# Integral of difference between PCG LFP

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import DataFrame
from scipy import signal
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.integrate import simps
from scipy import integrate
from scipy.linalg import dft
from scipy.interpolate import InterpolatedUnivariateSpline
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests


datana = []
clag = 'Gc'
#changepath
path = 'C:/Users/Wayne/Desktop/cvl/original_data/ALL'
os.chdir(path)
data = os.listdir(path)
for i in range(len(data)):
    d = data[i][12:17]
    datana.append(d)

dataname = [[0]]* len(data)
Goptimal = 'C:/Users/Wayne/Desktop/cvl/original_data/Gc.xlsx'
case_go = pd.read_excel(Goptimal)
case_go.head()
id = {j:[] for j in range(len(datana))}
for i in range(len(datana)):
    id[i] = datana[i]
iddf = DataFrame(list(id.values()),columns = ['Subject'])
Groups_subject_info = pd.merge(case_go, iddf,  left_on='Subject', right_on='Subject', how='left' )
Subjects = list(Groups_subject_info.Subject)
Groups = list(Groups_subject_info.Groups)
for k in range(len(Groups)):
    dataname[k] = 'data_signal_'+Subjects[k]+'_Gc.csv'

fs = 81920
samplinginterval = 1/fs
time = np.arange(0, 80000/fs, samplinginterval)
cutoff_high = 8
cutoff_low = 1
order = 2
dt=1/fs
#initialize spikes
spikes_r = {j:[] for j in range(len(dataname))}
spikes_l = {j:[] for j in range(len(dataname))}



def butter_smooth(order, cutoff_low, cutoff_high, data, fs):
    nyp = fs/2.0
    norm_cutoff_high = cutoff_high/nyp
    norm_cutoff_low = cutoff_low/nyp
    b,a = signal.butter(order, [norm_cutoff_low, norm_cutoff_high], 'bandpass', analog=False)
    y = signal.filtfilt(b, a, np.asarray(data))
    return y

def fir_bandpass(data, fs, cut_off_low, cut_off_high, width=2.0, ripple_db=10.0):
    nyq_rate = fs / 2.0
    wid = width/nyq_rate
    N, beta = signal.kaiserord(ripple_db, wid)
    taps = signal.firwin(N, cutoff = [cut_off_low, cut_off_high],
                  window = 'hamming', pass_zero = False, fs=fs)
    filtered_signal = signal.lfilter(taps, 1.0, data)
    return filtered_signal, N


# def butter_lowpass(order, cutoff, data, fs):
#     norm_cutoff = cutoff/nyp
#     b,a = butter(order, norm_cutoff, 'low', analog=False)
#     y = filtfilt(b, a, np.asarray(data))
#     return y


def slow_batch(data_r, data_l, na, group):

    fig_name = na+'_'+group

    PCG_R_filter, N = fir_bandpass(data=np.asarray(data_r),
                                   fs=fs, cut_off_low=cutoff_low,
                                   cut_off_high=cutoff_high, width=2 )
    PCG_L_filter, N = fir_bandpass(data=np.asarray(data_l),
                                   fs=fs, cut_off_low=cutoff_low,
                                   cut_off_high=cutoff_high, width=2 )
    delay = 0.5 * (N-1) / fs
    # PCG_R_filter = butter_smooth(order, cutoff_low, cutoff_high, data_r, fs)
    # PCG_L_filter = butter_smooth(order, cutoff_low, cutoff_high, data_l, fs)

    d1 = PCG_R_filter
    d2 = PCG_L_filter

    # x = np.linspace(-4*np.pi, 4*np.pi, 80000)
    # d1 = np.sin(x)
    # d2 = np.cos(x)

    y1_hilbert = signal.hilbert(d1)
    y2_hilbert = signal.hilbert(d2)

    y1_ang = np.angle(y1_hilbert)
    y2_ang = np.angle(y2_hilbert)

    y1_pow = abs(y1_hilbert)
    y2_pow = abs(y2_hilbert)

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    ang_r, _ = signal.find_peaks(smooth(y1_ang/np.pi, 100))
    ang_l, _ = signal.find_peaks(smooth(y2_ang/np.pi, 100))

    # low freq
    low_r, _ = signal.find_peaks(smooth(PCG_R_filter, 10), prominence=0.2)
    low_l, _ = signal.find_peaks(smooth(PCG_L_filter, 10), prominence=0.1)


    # high freq
    high_r, _ = signal.find_peaks(data_r, prominence=0.5)
    high_l, _ = signal.find_peaks(data_l, prominence=0.5)


### mark the valley point of the signal
    peak_ind_r, _ = signal.find_peaks(smooth(-PCG_R_filter, 10))
    peak_ind_l, _ = signal.find_peaks(smooth(-PCG_L_filter, 10))
    prominences_r = signal.peak_prominences(smooth(PCG_R_filter, 10), low_r)[0]
    prominences_l = signal.peak_prominences(smooth(PCG_L_filter, 10), low_l)[0]
    contour_heights_r = PCG_R_filter[low_r] - prominences_r
    contour_heights_l = PCG_L_filter[low_l] - prominences_l

    ### find the vally points in our signals
    b_low_r, _ = signal.find_peaks(smooth(-PCG_R_filter, 10), prominence=0.3, height= 1)
    b_low_l, _ = signal.find_peaks(smooth(-PCG_L_filter, 10), prominence=0.3, height= 1)
# print(Subjects[y],"_right_angle: ", np.round(ang_r/81920*2*180, 2))
# print(Subjects[y], "_left_angle: ", np.round(ang_l/81920*2*180, 2))


    # for i in range(len(y1_ang)):
    #     disc = (y1_ang[i]/np.pi) - (y2_ang[i]/np.pi)

    fig = plt.figure(figsize=(15, 10), dpi=100)
    fig.suptitle(fig_name, fontsize="x-large")
    a1 = plt.subplot(2,2,1)
    plt.plot(time, np.asarray(data_r),  alpha=0.7)
    plt.plot(time, np.asarray(data_l),  alpha=0.7)
    # plt.plot(time, smooth(PCG_R_filter, 10), label = "PCG_R_IIR")
    # plt.plot(time, smooth(PCG_L_filter, 10), label= "PCG_L_IIR")
    a_low_r, _ = signal.find_peaks(smooth(-PCG_R_filter[N-1:], 10), prominence=0.3, height= 1)
    a_low_l, _ = signal.find_peaks(smooth(-PCG_L_filter[N-1:], 10), prominence=0.3, height= 1)
    plt.plot(time[N-1:]-delay, smooth(PCG_R_filter[N-1:], 10), '#1f77b4', linewidth=3,label = "PCG_R")
    plt.plot(time[N-1:]-delay, smooth(PCG_L_filter[N-1:], 10), 'orange', linewidth=3,label= "PCG_L")
    plt.plot(b_low_r/fs - delay, smooth(PCG_R_filter, 10)[b_low_r], "or")
    plt.plot(b_low_l/fs - delay, smooth(PCG_L_filter, 10)[b_low_l], "ok")
    plt.plot(high_r/fs, data_r[high_r], "xr")
    plt.plot(high_l/fs, data_l[high_l], "xk")
    plt.title('PCG_original_time_series')
    plt.legend()



    plt.subplot(2,2,2, sharey = a1)
    plt.plot(time, smooth(PCG_R_filter, 10), label = "PCG_R")
    plt.plot(time, smooth(PCG_L_filter, 10), label= "PCG_L")
    plt.plot(low_r/fs, smooth(PCG_R_filter, 10)[low_r], "xr")
    plt.plot(low_l/fs, smooth(PCG_L_filter, 10)[low_l], "xk")
    plt.plot(b_low_r/fs, smooth(PCG_R_filter, 10)[b_low_r], "or")
    plt.plot(b_low_l/fs, smooth(PCG_L_filter, 10)[b_low_l], "ok")
    #plt.plot(peak_ind_r/fs, smooth(PCG_R_filter, 10)[peak_ind_r], "or")
    #plt.plot(peak_ind_l/fs, smooth(PCG_L_filter, 10)[peak_ind_l], "ok")
    plt.vlines(x=low_r/fs, ymin=contour_heights_r, ymax=PCG_R_filter[low_r])
    plt.vlines(x=low_l/fs, ymin=contour_heights_l, ymax=PCG_L_filter[low_l])
    plt.axhline(y=np.mean(contour_heights_r), linestyle='--')
    plt.axhline(y=np.mean(contour_heights_l), color = "orange",linestyle=':')


    #plt.fill_between(time, PCG_R_filter, PCG_L_filter, color='lightpink', alpha=0.4, hatch='|||||||')
    plt.title('PCG_low_pass_filter')

    plt.legend()
    #plt.stem(time, smooth(PCG_R_filter, 100), use_line_collection=True)
    #plt.stem(time, smooth(PCG_L_filter, 100))

    # subplot(3,2,3, sharey = a1)
    # plt.plot(time, np.real(y1_hilbert), label= "real_hilbert")
    # plt.plot(time, np.imag(y1_hilbert), label= "image_hilbert")
    # plt.title('Hilbert Transform_PCG_R')
    # plt.legend()
    #
    # subplot(3,2,4, sharey = a1)
    # plt.plot(time, np.real(y2_hilbert), label= "real_hilbert")
    # plt.plot(time, np.imag(y2_hilbert), label= "image_hilbert")
    # plt.title('Hilbert Transform_PCG_L')
    # plt.legend()

    plt.subplot(2, 2, 3, sharey=a1)
    plt.plot(time, smooth(y1_ang/np.pi, 100), label= "phase_PCG_R")
    plt.plot(time, smooth(y2_ang/np.pi, 100), label= "phase_PCG_L")
    plt.fill_between(time, smooth(y1_ang/np.pi, 100), smooth(y2_ang/np.pi, 100), color='lightpink', alpha=0.4, hatch='|||||||')
    plt.title("Instaneous Phase")
    #plt.legend(loc= "upper center")
    # plt.subplot(2, 2, 4)
    # plt.plot(time, y1_pow, label= "power_PCG_R")
    # plt.plot(time, y2_pow, label= "power_PCG_L")
    # plt.title("power")
    # plt.show()
    # plt.subplot(2, 2, 4, polar=True)
    # plt.plot(time*2*np.pi, smooth(y1_ang/np.pi, 100))
    # plt.plot(time*2*np.pi, smooth(y2_ang/np.pi, 100))
    # plt.show()
    plt.subplot(2, 2, 4)
    plt.plot(time, smooth(y1_ang/np.pi, 100) - smooth(y2_ang/np.pi, 100), label = "phase_lock")
    plt.title('Phase Lock')
    #plt.show()


    # ### low_amplitude
    # amp_r = [y*0 for y in range(len(b_low_r)-1)]
    # for i in range(len(b_low_r)-1):
    #     amp_r[i] = np.around(np.max(PCG_R_filter[b_low_r[i]:b_low_r[i+1]]) - np.mean(PCG_R_filter[b_low_r]), 2)
    # print("amp_r: ", amp_r)
    #
    #
    # if len(b_low_l) > 1:
    #     amp_l = [y*0 for y in range(len(b_low_l)-1)]
    #     for i in range(len(b_low_l)-1):
    #         amp_l[i] = np.around(np.max(PCG_L_filter[b_low_l[i]:b_low_l[i+1]]) - np.mean(PCG_L_filter[b_low_l]), 2)
    #     print("n_amp_l: ", amp_l)
    # else:
    #     amp_l = 0.0
    #     print("ab_amp_l: ", amp_l)


    ### high_amplitude
    amp_r = [y*0 for y in range(len(b_low_r)-1)]
    for i in range(len(b_low_r)-1):
        amp_r[i] = np.around(np.max(data_r[b_low_r[i]:b_low_r[i+1]])-np.mean(data_r[b_low_r]), 2)
    print("amp_r: ", amp_r)


    if len(b_low_l) >1:
        amp_l = [y*0 for y in range(len(b_low_l)-1)]
        for i in range(len(b_low_l)-1):
            amp_l[i] = np.around(np.max(data_l[b_low_l[i]:b_low_l[i+1]])-np.mean(data_l[b_low_l]), 2)
        print("n_amp_l: ", amp_l)
    else:
        amp_l = 0.0
        print("ab_amp_l: ", amp_l)





    # os.chdir("W:/lateralization_ppl/figs")
    # fig.savefig(group+'_' + na + '.png')
    # os.chdir(path)

    # raw_signal = {"right": data_r, "left": data_l}
    # df_raw = pd.DataFrame(raw_signal)
    # grangercausalitytests(x= df_raw[["right","left"]], maxlag=[600, 800])

###_________________________________________________
    #return [np.round(ang_r/81920*2*180, 2)], [np.round(ang_l/81920*2*180, 2)]
    #return [len(np.round(ang_r/81920*2*180, 2))], [len(np.round(ang_l/81920*2*180, 2))]
    #return [np.round(len(low_r), 2)], [np.round(len(low_l), 2)]
    #return [np.round(low_r/fs, 2)], [np.round(low_l/fs, 2)]
    return [amp_r], [amp_l]




###_________________________________________________
    # subplot(2, 2, 4, sharey=a1)
    # plt.plot(time, (y1_ang-y2_ang)/np.pi, label="phase_difference")
    # plt.show()

    # amp1 = np.sin(2 * np.pi * 4 * time)
    # amp2 = np.sin(-(2 * np.pi * 4 * time))

    # subplot(2, 2, 4)
    # plt.plot(time, amp1, label="phase_difference")
    # plt.plot(time, amp2, label="phase_difference")
    # plt.plot(time, amp1-amp2)

a1 = [[0]] * 74
b1 = [[0]] * 74


if __name__ == "__main__":
    for y in range(len(Subjects)):
        df = pd.read_csv(dataname[y])
        df.head()
        data_r = np.asarray(df.PCG_R)
        data_l = np.asarray(df.PCG_L)
        a, b = slow_batch(data_r, data_l, Subjects[y], Groups[y])
        a1[y] = a
        b1[y] = b
    ab = np.concatenate([a1, b1], axis=1)
    #hc = pd.DataFrame(ab, index=Subjects)
    hc = pd.DataFrame(ab, index=Groups)
    os.chdir("W:/lateralization_ppl/")
    hc.to_csv('ab_fast_amplitude.csv')





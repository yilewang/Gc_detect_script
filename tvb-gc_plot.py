import scipy.signal
from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import numpy as np
import tvb.simulator.plot.timeseries_interactive as ts_int
import warnings
from tvb.analyzers import fft
import matplotlib.pyplot as plt
import _pickle as cPickle
import pandas as pd
from matplotlib.pyplot import figure
from scipy.signal import savgol_filter
import xlrd

LOG = get_logger('demo')
warnings.filterwarnings('ignore')
case_go = pd.read_excel('C:/tvb3.0/Gc.xlsx')
case_go.head()
SNC_ID = case_go.values[0:10, 0]
NC_ID = case_go.values[10:26, 0]
MCI_ID = case_go.values[26:61, 0]
AD_ID = case_go.values[61:74, 0]
SNC_GO = case_go.values[0:10, 1]
NC_GO = case_go.values[10:26, 1]
MCI_GO = case_go.values[26:61, 1]
AD_GO = case_go.values[61:74, 1]

speed = 10.
data = []
t = []
data_branch = []
path = []
pth1 = 'C:/Users/Wayne/Desktop/AUS-76-DATASET/ALL_Structure_category/Joelle_normalized/'
#pth2 = 'SuperNormal/'
#pth2 = 'NormalControls/'
#pth2 = 'MCI/'
pth2 = 'AD/'
pth3 = '.zip'
peak = []

def csgo(file, go):
    my_rng = np.random.RandomState(seed=42)
    oscillator = ReducedSetHindmarshRose()
    white_matter = connectivity.Connectivity.from_file(file)
    oscillator.variables_of_interest = ["xi"]
    white_matter.speed = np.array([speed])
    white_matter_coupling = coupling.Linear(a=np.array([go]))
    heunint = integrators.HeunStochastic(dt=0.01220703125, noise=noise.Additive(nsig=np.array([0.00001]), ntau=0.0,
                                                                                random_stream=my_rng))
    mon_raw = monitors.Raw()
    mon_tavg = monitors.TemporalAverage(period=1.)
    what_to_watch = (mon_raw, mon_tavg)
    sim = simulator.Simulator(model=oscillator, connectivity=white_matter, coupling=white_matter_coupling,
                              integrator=heunint, monitors=what_to_watch)
    sim.configure()

    raw_data = []
    raw_time = []
    tavg_data = []
    tavg_time = []

    for raw, tavg in sim(simulation_length=1000):
        if not raw is None:
            raw_time.append(raw[0])
            raw_data.append(raw[1])

        if not tavg is None:
            tavg_time.append(tavg[0])
            tavg_data.append(tavg[1])

    # create and launch the interactive visualiser
    tsr = time_series.TimeSeriesRegion(data=np.array(raw_data), connectivity=sim.connectivity,
                                       sample_period=sim.monitors[0].period / 1e3,
                                       sample_period_unit='s')
    tsr.configure()
    tsr
    tsi = ts_int.TimeSeriesInteractive(time_series=tsr)
    tsi.configure()

    RAW = np.array(tsr.data)
    signal = RAW[1920:81920, 0, 5, 0]
    sm = savgol_filter(signal, 499, 2)
    H = np.mean(RAW[1920:81920, 0, 5, 0]) + 0.2
    peaks, _ = scipy.signal.find_peaks(sm, height=H)
    fil = scipy.signal.argrelmax(sm, order=199)
    if not peaks.any():
        final_peaks, _ = scipy.signal.find_peaks(sm, height=H)
        fil = []
        plt.plot(sm, label = 'PCG_R')
        plt.plot(final_peaks, sm[final_peaks], "x")
        plt.plot(np.zeros_like(sm), "--", color="gray")
        plt.legend(loc='upper right')
    else:
        peaks, _ = scipy.signal.find_peaks(sm, height=H)
        alls = []
        for i in range(len(peaks)):
            all_peaks = sm[peaks[i]]
            alls.append(all_peaks)
        G = H + ((np.max(alls) - H) / 2)
        final_peaks, _ = scipy.signal.find_peaks(sm, height=G)

        if np.array_equal(fil, final_peaks) == 1:
            fil = final_peaks
        else:
            fil = np.intersect1d(fil, final_peaks)
        AB = []
        ABC = []
        ABCD = []
        for i in range(len(fil)):
            ab = sm[fil[i]]
            abc = sm[fil[i] - 189]
            abcd = sm[fil[i] + 173]
            AB.append(ab)
            ABC.append(abc)
            ABCD.append(abcd)
            if abs(AB[i] - ABC[i]) < 0.11 or (abs(AB[i] - ABCD[i]) < 0.11):
                fil[i] = 0
            else:
                continue
        fil = np.delete(fil, np.where(fil == 0))
        plt.plot(sm, label = 'PCG_R')
        plt.plot(fil, sm[fil], "x")
        plt.plot(np.zeros_like(sm), "--", color="gray")
        plt.legend(loc = 'upper right')

    signal2 = RAW[1920:81920, 0, 4, 0]
    H2 = np.mean(RAW[1920:81920, 0, 4, 0]) + 0.2
    sm2 = savgol_filter(signal2, 499, 2)
    peaks2, _ = scipy.signal.find_peaks(sm2, height=H2)
    fil2 = scipy.signal.argrelmax(sm2, order=199)
    fil2 = fil2[0]
    if not peaks2.any():
        final_peaks2, _ = scipy.signal.find_peaks(sm2, height=H2)
        fil2 = []
        plt.plot(sm2, label = 'PCG_L')
        plt.plot(final_peaks2, sm2[final_peaks2], "x")
        plt.plot(np.zeros_like(sm2), "--", color="gray")
        plt.legend(loc='upper left')
    else:
        peaks2, _ = scipy.signal.find_peaks(sm2, height=H2)
        alls2 = []
        for i in range(len(peaks2)):
            all_peaks2 = sm2[peaks2[i]]
            alls2.append(all_peaks2)
        G2 = H2 + ((np.max(alls2) - H2) / 2)
        final_peaks2, _ = scipy.signal.find_peaks(sm2, height=G2)
        if np.array_equal(fil2, final_peaks2) == 1:
            fil2 = final_peaks2
        else:
            fil2 = np.intersect1d(fil2, final_peaks2)
        AB2 = []
        ABC2 = []
        ABCD2 = []
        for i in range(len(fil2)):
            ab2 = sm2[fil2[i]]
            abc2 = sm2[fil2[i] - 189]
            abcd2 = sm2[fil2[i] + 173]
            AB2.append(ab2)
            ABC2.append(abc2)
            ABCD2.append(abcd2)
            if abs(AB2[i] - ABC2[i]) < 0.11 or (abs(AB2[i] - ABCD2[i]) < 0.11):
                fil2[i] = 0
            else:
                continue
        fil2 = np.delete(fil2, np.where(fil2 == 0))
        plt.plot(sm2, label = 'PCG_L')
        plt.plot(fil2, sm2[fil2], "x")
        plt.plot(np.zeros_like(sm2), "--", color="gray")
        plt.legend(loc = 'upper left')
    return [fil, fil2]


for y in range(len(AD_ID)):
    pth = pth1 + pth2 + AD_ID[y] + pth3
    path.append(pth)
    figure(y)
    plt.ion()
    plt.title(AD_ID[y])
    num_peak = csgo(path[y], AD_GO[y])
    peak.append(num_peak)
    plt.savefig(str(AD_ID[y]))


plt.ioff()
plt.show()

# label = ['TPOmid_R', 'TPOmid_L', 'TPOsup_R', 'TPOsup_L', 'AMYG_R', 'AMYG_L', 'PHG_R', 'PHG_L', 'HIP_R', 'HIP_L', 'PCG_R', 'PCG_L', 'DCG_R', 'DCG_L', 'ACG_R', 'ACG_L']
# results = fft.FFT(time_series=tsr)
# eva = results.evaluate()
PCG_R = []
PCG_L = []
for i in range(len(peak)):
    PCGr = len(peak[i][0])
    PCGl = len(peak[i][1])
    PCG_R.append(PCGr)
    PCG_L.append(PCGl)
PCG = [PCG_R, PCG_L]
df = pd.DataFrame(PCG)
df.to_csv('num_spike_AD_Gc.csv')

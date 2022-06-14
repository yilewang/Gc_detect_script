#!/bin/usr/bin

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/mnt/w/github/tvbtools")
from tools.signalTools import freqCount, hdf5Reader

filename = "/mnt/w/go3mins/SNC/2820A.h5"
time = np.arange(0, 180, 1/81920)

dset = hdf5Reader(filename)

freqNum = freqCount(dset[:,5], 0.5, 81920., filter = True, visual=True, dpi = 30)
print(freqNum)
# pCNGRTheta, N, delay= fir_bandpass(dset[:,5], 81920., 2., 10.)
# fig, ax = plt.subplots(figsize=(100,15))
# ax.plot(time, dset[:, 5], label = "pCNGR-raw")
# ax.plot(time[N-1:]-delay, pCNGRTheta[N-1:], label = "pCNGR-Filter")
# ax.legend()
# plt.show()
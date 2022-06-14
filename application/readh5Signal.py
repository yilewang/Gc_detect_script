#!/bin/usr/bin

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/mnt/c/Users/wayne/tvb/tvbtools")
from tools.signal import fir_bandpass

filename = "/mnt/c/Users/wayne/tvb/gc3mins/AD/0306A.h5"
with h5py.File(filename, "r") as f:
    ### get key of h5 file
    # print(list(f.keys()))
    time = np.arange(0, 180, 1/81920)
    dset = f['raw']
    pCNGRTheta, N, delay= fir_bandpass(dset[:,5], 81920., 2., 10.)
    fig, ax = plt.subplots(figsize=(100,15))
    ax.plot(time, dset[:, 5], label = "pCNGR-raw")
    ax.plot(time[N-1:]-delay, pCNGRTheta[N-1:], label = "pCNGR-Filter")
    ax.legend()
    plt.show()
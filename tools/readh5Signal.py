#!/usr/bin/python

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import FIR_filter

filename = "C:/Users/wayne/tvb/gc3mins/AD/0306A.h5"
with h5py.File(filename, "r") as f:
    ### get key of h5 file
    # print(list(f.keys()))
    time = np.arange(0, 180, 1/81920)
    dset = f['raw']
    pCNGRTheta, N, delay= FIR_filter.fir_bandpass(dset[:,5], 81920., 2., 10.)
    fig, ax = plt.subplots(figsize=(100, 15))
    ax.plot(time[:81920*90], dset[:81920*90,5], label = "pCNGR-raw")
    ax.plot(time[:81920*90][N-1:]-delay, pCNGRTheta[:81920*90][N-1:], label = "pCNGR-Filter")
    ax.legend()
    plt.show()
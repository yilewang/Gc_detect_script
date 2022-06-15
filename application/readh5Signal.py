#!/bin/usr/bin

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/mnt/w/github/tvbtools")
from tools.signalTools import freqCount, hdf5Reader

filename = "/mnt/w/go3mins/SNC/2820A.h5"
dset = hdf5Reader(filename)

freqNum = freqCount(dset[:,5], 0.5, 81920., filter=False, visual=True, dpi = 30)
print(freqNum)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pactools import Comodulogram, REFERENCES
from pactools import simulate_pac
import argparse
import time
# import h5py
# parser = argparse.ArgumentParser(description='pass a str')
# parser.add_argument('--path',type=str, required=True, help='read_data')
# args = parser.parse_args()

# path = args.path 


# with h5py.File(path, "r") as f:
#     ### get key of h5 file
#         key = list(f.keys())
#         dset = f[key[0]][:]

# data = 

# signal_left = data['pCNG-L'].to_numpy()
# signal_right = data['pCNG-R'].to_numpy()

# def comodulograms(signal, fs=1024, low_fq_width=1.0, low_fq_range=np.linspace(1,10,50), methods='tort'):
#     # Compute the comodulograms and plot them
#     print('%s... ' % (methods, ))
#     estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
#                                 low_fq_width=low_fq_width, method=methods,
#                                 progress_bar=False)
#     estimator.fit(signal)
#     estimator.plot(titles=)

# comodulograms(signal_left)
# comodulograms(signal_right)

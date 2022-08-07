# ########
# # test #
# ########

# from asyncore import read
# import numpy as np
# import matplotlib.pyplot as plt

# from pactools import Comodulogram, REFERENCES
# from pactools import simulate_pac
# from tools.signaltools import SignalToolkit
# fs = 200.  # Hz
# high_fq = 50.0  # Hz
# low_fq = 5.0  # Hz
# low_fq_width = 1.0  # Hz

# n_points = 10000
# noise_level = 0.4

# signala = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
#                       low_fq_width=low_fq_width, noise_level=noise_level,
#                       random_state=0)

# SignalToolkit.PAC_comodulogram(signala, [2,20,2],[30, 90, 10], fs=200)

# # signala = df_left

# low_fq_range = np.linspace(1, 10, 50)
# methods = [
#     'ozkurt', 'canolty', 'tort', 'penny', 'vanwijk', 'duprelatour', 'colgin',
#     'sigl', 'bispectrum'
# ]


# # Define the subplots where the comodulogram will be plotted
# n_lines = 3
# n_columns = int(np.ceil(len(methods) / float(n_lines)))
# fig, axs = plt.subplots(
#     n_lines, n_columns, figsize=(4 * n_columns, 3 * n_lines))
# axs = axs.ravel()


# # Compute the comodulograms and plot them
# for ax, method in zip(axs, methods):
#     print('%s... ' % (method, ))
#     estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
#                              low_fq_width=low_fq_width, method=method,
#                              progress_bar=False)
#     estimator.fit(signala)
#     estimator.plot(titles=[REFERENCES[method]], axs=[ax])

# plt.show()

import pandas as pd
import shutil
import numpy as np
path = '/mnt/c/Users/Wayne/tvb/stat_data/Gc_Go.xlsx'
coData = pd.read_excel(path)

def read_data(grp, caseid):
    gm = np.round(coData.loc[coData['caseid'] == caseid, 'Gc'].item(), 3)
    filename = '/mnt/d/data/LFP/'+str(grp)+'/'+str(caseid)+'/'+str(caseid)+'_'+str(gm)+'.csv'
    destin = '/home/yat-lok/workspace/data4project/lateralization/LFP_critical/' + str(grp)+'/'+str(caseid)+'_'+str(gm)+'.csv'
    shutil.copyfile(filename, destin)

for grp, caseid in zip(coData.groups, coData.caseid):
    read_data(grp, caseid)
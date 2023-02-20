# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from pactools import Comodulogram, REFERENCES
# import pandas as pd
# import seaborn as sns
# from signaltools import SignalToolkit
# df = pd.read_excel('/Users/yat-lok/workspaces/data4project/mega_table.xlsx', sheet_name='tvb_parameters')
# groups = ['SNC', 'NC', 'MCI', 'AD']

# def comodulograms(signal, fs=1024, low_fq_width=1.0, low_fq_range=np.linspace(1,10,50), methods='tort'):
#     # Compute the comodulograms and plot them
#     print('%s... ' % (methods, ))
#     estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
#                                 low_fq_width=low_fq_width, method=methods,
#                                 progress_bar=False)
#     estimator.fit(signal)
#     estimator.plot()
#     plt.show()
#     return estimator.comod_





# # signal_left_filtered = SignalToolkit.sos_filter(signal_left, [2,100], fs=81920)
# # signal_right_filtered = SignalToolkit.sos_filter(signal_right, [2,100], fs=81920)
# mat = comodulograms(signal_left)
# fig, ax = plt.subplots()
# ax = sns.heatmap(mat.T)
# ax.invert_yaxis()


    



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate two sample signals with different frequency content
# t = np.linspace(0, 1, 1000)
# x1 = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
# x2 = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t)

# path = '/Users/yat-lok/workspaces/data4project/lateralization/LFP_critical/AD/0306A_0.015.csv'
# data = pd.read_csv(path)
# x1 = data['pCNG-L']
# x2 = data['pCNG-R']
# t = np.linspace(0,1,81920)

# # Compute the FFT of the two signals
# X1 = np.fft.fft(x1)
# X2 = np.fft.fft(x2)

# # Compute the magnitudes of the FFT coefficients
# mag1 = np.abs(X1)
# mag2 = np.abs(X2)

# # Calculate the difference in magnitudes at each frequency bin
# diff = mag1 - mag2

# # Plot the magnitude spectra and the difference
# freqs = np.fft.fftfreq(len(t), t[1] - t[0])
# plt.plot(freqs, mag1, label='Signal 1')
# plt.plot(freqs, mag2, label='Signal 2')
# plt.plot(freqs, diff, label='Difference')
# plt.legend()
# plt.show()




# import pywt
# import numpy as np
# import matplotlib.pyplot as plt

# # Generate two sample signals with different frequency content

# path = '/Users/yat-lok/workspaces/data4project/lateralization/LFP_critical/AD/0306A_0.015.csv'
# data = pd.read_csv(path)
# x1 = data['pCNG-L']
# x2 = data['pCNG-R']
# t = np.linspace(0,1,81920)


# # t = np.linspace(0, 1, 1000)
# # x1 = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
# # x2 = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t)

# # Choose a wavelet function and a decomposition level
# wavelet = 'db4'
# level = 5

# # Apply the wavelet transform to both signals
# coeffs1 = pywt.wavedec(x1, wavelet, level=level)
# coeffs2 = pywt.wavedec(x2, wavelet, level=level)

# # Calculate the magnitudes of the wavelet coefficients
# mag1 = [np.abs(c) for c in coeffs1]
# mag2 = [np.abs(c) for c in coeffs2]

# # Calculate the difference in magnitudes at each decomposition level
# diff = [np.abs(m1 - m2) for m1, m2 in zip(mag1, mag2)]

# # Plot the wavelet coefficients and the difference
# plt.figure(figsize=(8, 6))
# for i in range(level):
#     plt.subplot(level, 2, 2 * i + 1)
#     plt.plot(mag1[i], label='Signal 1')
#     plt.plot(mag2[i], label='Signal 2')
#     plt.title('Level {}'.format(i))
#     plt.legend()
#     plt.subplot(level, 2, 2 * i + 2)
#     plt.plot(diff[i], label='Difference')
#     plt.legend()
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# Generate two sample signals
t = np.linspace(0, 10, 1000)
x1 = np.sin(2 * np.pi * 5 * t)
x2 = np.sin(2 * np.pi * 5 * t + np.pi/4)

path = '/Users/yat-lok/workspaces/data4project/lateralization/LFP_critical/AD/0506A_0.019.csv'
data = pd.read_csv(path)
x1 = data['pCNG-L']
x2 = data['pCNG-R']
t = np.linspace(0,1,81920)


# Compute their correlation coefficient
corr_coef = np.corrcoef(x1, x2)[0, 1]
print('Correlation coefficient:', corr_coef)

# Compute their cross-correlation function
# cross_corr = np.correlate(x1, x2, mode='full')
# lags = np.arange(-len(x1) + 1, len(x1))
# plt.figure()
# plt.plot(lags, cross_corr)
# plt.xlabel('Lag')
# plt.ylabel('Cross-correlation')
# plt.title('Cross-correlation function')
# plt.show()

# Compute Granger causality
x1 = np.array(x1)
x2= np.array(x2)
data = np.column_stack((x1, x2))
gc_res = grangercausalitytests(data, 1)

inner_keys = list(gc_res[list(gc_res)[0]][0])

pvalues = [gc_res[list(gc_res)[0]][0][i][1] for i in inner_keys]
print('Granger causality p-values:', pvalues)

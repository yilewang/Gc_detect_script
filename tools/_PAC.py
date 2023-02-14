import signaltools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pactools import Comodulogram, REFERENCES
from pactools import simulate_pac

# fs = 200.  # Hz
# high_fq = 50.0  # Hz
# low_fq = 5.0  # Hz
# low_fq_width = 1.0  # Hz

# n_points = 10000
# noise_level = 0.4

# signal = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
#                       low_fq_width=low_fq_width, noise_level=noise_level,
#                       random_state=0)


data = pd.read_csv('/Users/yat-lok/workspaces/data4project/lateralization/LFP_critical/NC/0705A_0.016.csv')

signal_left = data['pCNG-L'].to_numpy()
signal_right = data['pCNG-R'].to_numpy()

def comodulograms(signal, fs=1024, low_fq_width=1.0, low_fq_range=np.linspace(1,10,50), methods='tort'):
    # Compute the comodulograms and plot them
    print('%s... ' % (methods, ))
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                low_fq_width=low_fq_width, method=methods,
                                progress_bar=False)
    estimator.fit(signal)
    estimator.plot(titles=[REFERENCES[methods]])
    plt.show()

comodulograms(signal_left)
comodulograms(signal_right)

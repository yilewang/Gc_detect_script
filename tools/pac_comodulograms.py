import h5py
import numpy as np
import matplotlib.pyplot as plt
from pactools import Comodulogram, REFERENCES


def comodulograms(signal, fs=1024, low_fq_width=1.0, low_fq_range=np.linspace(1,10,50), methods='tort'):
    # Compute the comodulograms and plot them
    print('%s... ' % (methods, ))
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                low_fq_width=low_fq_width, method=methods,
                                progress_bar=False)
    estimator.fit(signal)
    estimator.plot()
    plt.show()
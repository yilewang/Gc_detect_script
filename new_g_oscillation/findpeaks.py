#!/bin/usr/python
"""
functions to detect peaks
"""


import numpy as np
import matplotlib.pyplot as plt

class PeakFinder:
    def __init__(self, data):
        self.signal = np.array(data)

    def findPeaks(self):
        """
        It is a new self-designed python function to detect peaks and valleys points
        in 1D local field potentials data, especially for slow waves(Theta band)
        @Author: Yile Wang
        @Data: 05/29/2021
        @Email: yile.wang@utdallas.edu

        """
        Num = len(self.signal)
        Range = np.arange(0, Num, 1)
        pro = 0
        neg = 0
        ini = self.signal[0]
        Gp = []
        Gv = []
        for i in Range[:-1]:
            while self.signal[i] > ini:
                pro += self.signal[i] - ini
                if self.signal[i] > self.signal[i+1]:
                    Gp.append(i)
                ini = self.signal[i]
            while self.signal[i] < ini:
                neg += ini - self.signal[i]
                if self.signal[i] <= self.signal[i+1]:
                    Gv.append(i)
                ini = self.signal[i]
        return np.array(Gp), np.array(Gv)
    
    def peaks_plots(self, Gp, Gv):
        """
        A simple visulization tool for validity check
        Gp = array of peaks index
        Gv = array of valleys index
        Usage:
            data = [1,2,3,4,5,0,1,23,4,10, 20, 3, 40, 38, 2,3,4,6,0]
            a = PeakFinder(data)
            tmp1, tmp2 = a.findPeaks()
            a.peaks_plots(tmp1, tmp2) # for validity check
        """
        plt.figure()
        plt.title("Simple Visualization Tool for Peaks/Valleys Detection Check")
        plt.plot(self.signal, label = "raw signal")
        plt.plot(Gp, self.signal[Gp], 'x:r',label = "peaks points")
        plt.plot(Gv, self.signal[Gv], '*:g',label = "valleys poitns")
        plt.legend()
        plt.show()





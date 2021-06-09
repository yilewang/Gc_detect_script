#!/bin/usr/python
"""
functions to detect peaks
@Author: Yile Wang
@Data: 05/29/2021
@Email: yile.wang@utdallas.edu
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

        """
        Num = len(self.signal)
        Range = np.arange(0, Num, 1)
        ini = self.signal[0]
        
        # some elements
        Gp = [] # for peak point
        Gv = [] # for valley point

        # to get all the local maxima and minima
        for i in Range[:-1]:
            while self.signal[i] > ini:
                if self.signal[i] > self.signal[i+1]:
                    Gp.append(i)
                ini = self.signal[i]
            while self.signal[i] < ini:
                if self.signal[i] <= self.signal[i+1]:
                    Gv.append(i)
                ini = self.signal[i]

        # merge the peaks and valleys points
        Gall = np.sort(np.concatenate((Gp, Gv), axis = None))

        """
        Using slide windows to make second judgement about the peaks and valleys.
        All the signal can be decomposited as a V shape for further analysis
        """
        fs = 81920
        tmpGp = [x for x in Gp if x > 0.3*fs]
        tmpGv = [x for x in Gv if x > 0.3*fs]
        tmpRange = np.average(self.signal[tmpGp]) - np.average(self.signal[tmpGv])
        n = 2
        pks = np.array([])
        if tmpRange >= 0.1:
            Gap = np.average(self.signal[tmpGv]) + ((np.average(self.signal[tmpGp]) - np.average(self.signal[tmpGv]))/3)
            bottomIndex = Gall[self.signal[Gall] < Gap]
            for i in range(0, len(bottomIndex)-n+1, 1):
                tmp_windows = bottomIndex[i:i+n]
                tmp = [x for x in Gp if tmp_windows[0] <= x <= tmp_windows[1]]
                pks = np.append(pks, np.average(tmp))
            pks = pks.astype(int)
        return pks



    
    def peaks_plots(self, G):
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
        plt.plot(G, self.signal[G], 'x:r',label = "peaks and valleys points")
        plt.legend()
        plt.show()


# data = [5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
# func1 = PeakFinder(data)
# bb = func1.findPeaks()
# func1.peaks_plots(bb)

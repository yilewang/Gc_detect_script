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
        pro = 0 # for positive salience
        neg = 0 # for the negative salience
        proList = [] # positive salience collection
        negList = [] # negative salience collection

        # to get all the local maxima and minima
        for i in Range[:-1]:
            while self.signal[i] > ini:
                pro += self.signal[i] - ini
                if self.signal[i] > self.signal[i+1]:
                    Gp.append(i)
                    proList.append(np.round(pro, 5))
                    pro = 0
                ini = self.signal[i]
            while self.signal[i] < ini:
                neg += ini - self.signal[i]
                if self.signal[i] <= self.signal[i+1]:
                    Gv.append(i)
                    negList.append(np.round(neg, 5))
                    neg = 0
                ini = self.signal[i]

        # merge the peaks and valleys points
        Gall = np.sort(np.concatenate((Gp, Gv), axis = None))

        # using the slide windows to drop some unqualified peaks and valleys
        n = 3

        # the value for salience value
        value = np.var(self.signal[5000:])*2

        """
        Using slide windows to make second judgement about the peaks and valleys.
        All the signal can be decomposited as a V shape for further analysis
        """
        NewG = [Gall[0]]
        for i in range(1, len(Gall)-n+1, 2):
            tmp_windows = self.signal[Gall[i:i+n]]
            space01 = tmp_windows[0] - tmp_windows[1]
            space21 = tmp_windows[2] - tmp_windows[1]

            # condition1, both sides are less than value
            if space01 < value and space21 < value:
                if i ==1:
                    NewG.append(Gall[i])

            # condition2, left side is bigger, right side is smaller
            elif space01 > value and space21 < value:
                while i == 1:
                    NewG.append(Gall[i])
                    break
                if np.abs(space21) < np.abs(space01)/2 and self.signal[Gall[i+1]] < -1:
                    NewG.append(Gall[i+1])

            # condition3, left side is smaller, righ side is bigger
            elif space01 < value and space21 > value:
                if i == 1:
                    NewG.append(Gall[i])
                    if np.abs(space21) > np.abs(space01)*2:
                        NewG.append(Gall[i+1])
                        NewG.append(Gall[i+2])
                if np.abs(space01) < np.abs(space21)/2 and self.signal[Gall[i+1]] < -1:
                    NewG.append(Gall[i+2])

            # condition4, both sides are bigger
            elif space01 > value and space21 > value:
                while i == 1:
                    NewG.append(Gall[1])
                    break
                NewG.append(Gall[i+1])
                NewG.append(Gall[i+2])
        return np.array(NewG)

    
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





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

"""
@Author: Yile Wang

This is an easy function to read and calculate the correlation across brain regions
"""
class ReadRaw:
    def __init__(self, g, id, path):
        self.globalCoupling = g
        self.caseID = id
        self.filePath = path

    def signal_plot(self):
       df = pd.read_csv(self.filePath, index_col=0)
       plt.figure()


    def read_rawMatrices(self):
        df = pd.read_csv(self.filePath, index_col=0)
        corrMatrix = df.corr()
        return corrMatrix


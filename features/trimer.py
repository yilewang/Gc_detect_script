#!/usr/bin/python

import pandas as pd
import numpy as np
import sys
#sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\new_g_optimal')
sys.path.append('/home/wayne/github/TVB_workflow/new_g_optimal')
print(sys.path)
from read_mat import Case
from read_corrMatrix import ReadRaw
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
import itertools
import os

"""
@Author: Yile Wang

This script is used to calculate the homotopic meta-connectivity in four groups, SNC, NC, MCI, AD
"""

# brain region labels for your reference
regions = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R']
groups = ['SNC', 'NC', 'MCI','AD']

# iterate simulated functional connectivity


if __name__ == "__main__":
    for grp in groups:
        ldir = os.listdir('/home/wayne/TS-4-Vik/'+grp+'-TS/')
        for y in ldir:
            corrResult = []
            # import empirical functional connectivity
            try:
                # Here is the path of the mat file of the FC data
                pth_efc = "/home/wayne/TS-4-Vik/"+grp+"-TS/"+ y +"/ROISignals_"+ y +".mat"
                a2 = Case(pth_efc)
                df2 = pd.DataFrame.from_dict(a2.readFile().get("ROISignals"))
                df2.columns = regions

            except:
                continue


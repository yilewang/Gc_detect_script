#!/usr/bin/python

import pandas as pd
import numpy as np
import sys
sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\new_g_optimal')
sys.path.append('C:\\Users\\Wayne\\tvb\\Network-science-Toolbox\\Python')
# sys.path.append('/home/wayne/github/TVB_workflow/new_g_optimal')
# sys.path.append('/home/wayne/github/Network-science-Toolbox/Python')
from TS2dFCstream import TS2dFCstream
from dFCstream2Trimers import dFCstream2Trimers
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
    Trimer_Results = pd.DataFrame(columns=['grp','caseid', 'trimer_results','aCNG','mCNG','pCNG','HIP','PHG','AMY','sTEMp','mTEMp' ])
    Trimer_Heter  = pd.DataFrame(columns=['grp','caseid', 'trimer_results','aCNG','mCNG','pCNG','HIP','PHG','AMY','sTEMp','mTEMp' ])

    for grp in groups:
        # subject case ids
        ldir = os.listdir("C:/Users/Wayne/tvb/TS-4-Vik/"+ grp+'-TS')
        # ldir = os.listdir('/home/wayne/TS-4-Vik/'+grp+'-TS/')
        for y in ldir:
            # import empirical functional connectivity
            # Here is the path of the mat file of the FC data
            pth_efc = "C:/Users/Wayne/tvb/TS-4-Vik/"+ grp+'-TS/'+ y +"/ROISignals_"+ y +".mat"
            # pth_efc = "/home/wayne/TS-4-Vik/"+grp+"-TS/"+ y +"/ROISignals_"+ y +".mat"
            a2 = Case(pth_efc)
            df2 = pd.DataFrame.from_dict(a2.readFile().get("ROISignals"))
            df2.columns = regions

            # calculate the meta-connectivity, using existing script:
            dFCstream = TS2dFCstream(df2.to_numpy(), 5, None, '2D')
            MC_Trimers = dFCstream2Trimers(dFCstream)
            # do the averaging in the dimension3
            MC_avg = np.mean(MC_Trimers, 2) #n x n 
            tmp_trimer = np.array([])
            tmp_heter = np.array([])
            # pick up homotopic connection
            for i in range(0,15,2):
                j = i+1
                homotopic = MC_avg[i,j]
                heter_list = list(range(1,16,2))
                heter_list.remove(j)
                hetertopic = []
                for x in heter_list:
                    hetertopic.append(MC_avg[i,x])
                tmp_heter = np.append(tmp_heter, np.mean(hetertopic))
                tmp_trimer = np.append(tmp_trimer, homotopic)
            Trimer_Results = Trimer_Results.append({'grp':grp,'caseid':y,'trimer_results':np.mean(tmp_trimer), 'aCNG':tmp_trimer[0],'mCNG':tmp_trimer[1],'pCNG':tmp_trimer[2],'HIP':tmp_trimer[3],'PHG':tmp_trimer[4],  'AMY':tmp_trimer[5],'sTEMp':tmp_trimer[6],'mTEMp':tmp_trimer[7]},ignore_index=True)
            Trimer_Heter = Trimer_Heter.append({'grp':grp,'caseid':y,'trimer_results':np.mean(tmp_heter), 'aCNG':tmp_heter[0],'mCNG':tmp_heter[1],'pCNG':tmp_heter[2],'HIP':tmp_heter[3],'PHG':tmp_heter[4],'AMY':tmp_heter[5],'sTEMp':tmp_heter[6],'mTEMp':tmp_heter[7]},ignore_index=True)


    sns.set_theme(style="whitegrid")
    for i in list(Trimer_Results.columns[2:]):
        fig = plt.figure(figsize=(10,10))
        plt.title(f'Homotopic: {i}')
        fig = sns.violinplot(x="grp", y=i, data=Trimer_Results, capsize=.2,palette=["#66CDAA","#4682B4","#AB63FA","#FFA15A"])
        fig = sns.stripplot(x="grp", y=i, data=Trimer_Results,color='black')
        fig = sns.pointplot(data=Trimer_Results, x='grp', y=i, join=False, ci=None, color='red')
        fig.set_ylim(-0.5, 1)
        fig.set_yticks(np.arange(-0.5, 1, 0.1))
        tmp_name = [i,'.png']
        name = ''.join(tmp_name)
        plt.savefig(name)


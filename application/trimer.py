#!/usr/bin/python
from turtle import color
import pandas as pd
import numpy as np
import sys
sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\new_g_optimal')
sys.path.append('C:\\Users\\Wayne\\tvb\\Network-science-Toolbox\\Python')
# sys.path.append('/home/wayne/github/TVB_workflow/new_g_optimal')
# sys.path.append('/home/wayne/github/Network-science-Toolbox/Python')
from TS2dFCstream import TS2dFCstream
from dFCstream2Trimers import dFCstream2Trimers
from dFCstream2MC import dFCstream2MC
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
regionswithgroups = ['groups','aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R']
groups = ['SNC', 'NC', 'MCI','AD']
regionsHalf = np.array(['aCNG', 'mCNG','pCNG','HIP','PHG','AMY','sTEMp', 'mTEMp'])

regions14 = []
for i in range(14):
    wt = ["regions_", str(i)]
    wt = "".join(wt)
    regions14.append(wt)

# iterate simulated functional connectivity
if __name__ == "__main__":
    Trimer_Homo = pd.DataFrame(columns=['groups','trimer_homo','aCNG','mCNG','pCNG','HIP','PHG','AMY','sTEMp','mTEMp' ])
    Trimer_Hetero  = pd.DataFrame(columns=['groups','trimer_hetero','aCNG','mCNG','pCNG','HIP','PHG','AMY','sTEMp','mTEMp' ])
    Trimer = pd.DataFrame()
    
    for grp in groups:
        # subject case ids
        ldir = os.listdir("C:/Users/Wayne/tvb/TS-4-Vik/"+ grp+'-TS')
        # ldir = os.listdir('/home/wayne/TS-4-Vik/'+grp+'-TS/')
        MC_all = np.zeros((16,16, 16, len(ldir)))
        tmp_homo = np.array([])
        homoRegions = np.ones((1,len(regionsHalf)))
        for ind, y in enumerate(ldir):
            # import empirical functional connectivity
            # Here is the path of the mat file of the FC data
            pth_efc = "C:/Users/Wayne/tvb/TS-4-Vik/"+ grp+'-TS/'+ y +"/ROISignals_"+ y +".mat"
            # pth_efc = "/home/wayne/TS-4-Vik/"+grp+"-TS/"+ y +"/ROISignals_"+ y +".mat"
            a2 = Case(pth_efc)
            df2 = pd.DataFrame.from_dict(a2.readFile().get("ROISignals"))
            df2.columns = regions
            # calculate the meta-connectivity, using existing script:
            dFCstream = TS2dFCstream(df2.to_numpy(), 5, None, '2D')
            # Calculate MC
            MC_MC = dFCstream2MC(dFCstream)
            # Calculate Trimers results, with nxnxn information
            MC_Trimers = dFCstream2Trimers(dFCstream)
            MC_all[:,:,:,ind] = MC_Trimers
        MC_homo = np.mean(MC_all, 3)
        MC_single_groups = np.zeros((14, 8))
        # only pick up L sides of the regions
        for idnx, i in enumerate(range(0,15,2)):
            j = i+1 # represent R side
            newList = list(range(16))
            del newList[i:j+1] # drop the target regions L and R
            for idx, restNode in enumerate(newList):
                MC_single_groups[idx,idnx] = MC_homo[i,j,restNode] # In rest of the 14 regions, iternate the third dimensions, and pick up the homotopic MC
        MC_single = pd.DataFrame(MC_single_groups, index=regions14, columns=regionsHalf)
        grpInfo = [grp] * 14
        MC_single.insert(0, "groups", grpInfo)
        Trimer = Trimer.append(MC_single)
    print(Trimer)
    Trimer.to_excel("MC_homo.xlsx")
        #     Trimer = Trimer.append({'groups':grp,'aCNG':tmp_homo[0],'mCNG':tmp_homo[1],'pCNG':tmp_homo[2],'HIP':tmp_homo[3],'PHG':tmp_homo[4],  'AMY':tmp_homo[5],'sTEMp':tmp_homo[6],'mTEMp':tmp_homo[7]}, ignore_index=True)
        # print(Trimer)


            # MC_avg = np.mean(MC_Trimers,2)
            # tmp_homo = np.array([])
            # tmp_heterL = np.array([])
            # tmp_heterR = np.array([])
            # # pick up homotopic connection
            # for i in range(0,15,2):
            #     j = i+1
            #     homotopic = MC_avg[i,j]
            #     heter_list = list(range(1,16,2))
            #     heter_list.remove(j)
            #     hetertopic = []
            #     for x in heter_list:
            #         hetertopic.append(MC_avg[i,x])
            #     tmp_heterL = np.append(tmp_heterL, np.mean(hetertopic))
            #     tmp_homo = np.append(tmp_homo, homotopic)

            # for i in range(1,16,2):
            #     j = i-1
            #     heter_list = list(range(0,15,2))
            #     heter_list.remove(j)
            #     hetertopic = []
            #     for x in heter_list:
            #         hetertopic.append(MC_avg[i,x])
            #     tmp_heterR = np.append(tmp_heterR, np.mean(hetertopic))

            # Trimer_Homo = Trimer_Homo.append({'grp':grp,'caseid':y,'trimer_homo':np.mean(tmp_homo), 'aCNG':tmp_homo[0],'mCNG':tmp_homo[1],'pCNG':tmp_homo[2],'HIP':tmp_homo[3],'PHG':tmp_homo[4],  'AMY':tmp_homo[5],'sTEMp':tmp_homo[6],'mTEMp':tmp_homo[7]},ignore_index=True)

            # Trimer_Hetero = Trimer_Hetero.append({'grp':grp,'caseid':y,'trimer_hetero':(np.mean(tmp_heterL) + np.mean(tmp_heterR))/2, 'aCNG':(tmp_heterL[0]+tmp_heterR[0])/2,'mCNG':(tmp_heterL[1]+tmp_heterR[1])/2,'pCNG':(tmp_heterL[2]+tmp_heterR[2])/2,'HIP':(tmp_heterL[3]+tmp_heterR[3])/2,'PHG':(tmp_heterL[4]+tmp_heterR[4])/2,'AMY':(tmp_heterL[5]+tmp_heterR[5])/2,'sTEMp':(tmp_heterL[6]+tmp_heterR[6])/2,'mTEMp':(tmp_heterL[7]+tmp_heterR[7])/2},ignore_index=True)
    #     MC_grp_avg = np.mean(MC_heat, 2)
    #     MC_grp_avg = pd.DataFrame(MC_grp_avg, index=regions, columns=regions)
    #     # plt.figure(figsize=(10,7))
    #     # sns.set_theme()
    #     # plt.title(grp)
    #     # g = sns.heatmap(MC_grp_avg, cmap="coolwarm")
    #     # g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right',fontweight='light')
    #     # plt.savefig(f"{grp}.png", dpi=300)
    # Trimer_Homo.to_excel("meta.xlsx")

    # sns.set_theme(style="whitegrid")
    # for i in list(Trimer_Results.columns[2:]):
    #     fig = plt.figure(figsize=(10,10))
    #     plt.title(f'Homotopic: {i}')
    #     fig = sns.violinplot(x="grp", y=i, data=Trimer_Results, capsize=.2,palette=["#66CDAA","#4682B4","#AB63FA","#FFA15A"])
    #     fig = sns.stripplot(x="grp", y=i, data=Trimer_Results,color='black')
    #     fig = sns.pointplot(data=Trimer_Results, x='grp', y=i, join=False, ci=None, color='red')
    #     fig.set_ylim(-0.5, 1)
    #     fig.set_yticks(np.arange(-0.5, 1, 0.1))
    #     tmp_name = [i,'.png']
    #     name = ''.join(tmp_name)
    #     plt.savefig(name)


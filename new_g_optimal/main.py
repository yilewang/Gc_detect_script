
import pandas as pd
import numpy as np
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

This script is used for detecting G optimal point
The G optimal point, bascially it is the highest correlation coeficient 
between simulated functional connectivity and empirical functional connectivity
"""

# brain region labels for your reference
regions = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R']
groups = ['SNC', 'NC', 'MCI','AD']

# iterate simulated functional connectivity


if __name__ == "__main__":
    for grp in groups:
        ldir = os.listdir('C:/Users/Wayne/output/'+grp+'/')
        Goptimal = []
        tmp = np.round(np.arange(0.01, 0.061, 0.001),3)
        # a new dataframe to store the data
        df = pd.DataFrame(columns = ['caseid','new_Go'])
        for y in ldir:
            corrResult = []
            # import empirical functional connectivity
            try:
                # Here is the path of the mat file of the FC data
                pth_efc = "C:/Users/Wayne/workdir/TS-4-Vik/"+grp+"-TS/"+ y +"/ROICorrelation_"+ y +".mat"
                a2 = Case(pth_efc)
                df2 = pd.DataFrame.from_dict(a2.readFile().get("ROICorrelation"))
                df2.columns = regions
                df2.index = regions
                # G range is from 0.01 to 0.08
                for i in np.round(np.arange(0.01, 0.080, 0.001), 3):
                    # I already store all the simulated data in local path
                    pth = 'C:/Users/Wayne/output/'+grp+'/'+y+'/'+ y +'_'+str(i)+'.csv'
                    a1=ReadRaw(i, y, pth)
                    df1=a1.read_rawMatrices()
                    # keep half of the correlation matrices
                    sfc = np.tril(df1, -1)
                    efc = np.tril(df2, -1)
                    # break down them for correlation analysis
                    sfc_l = np.array([list(itertools.chain.from_iterable(np.ndarray.tolist(sfc)))])
                    sfc_array = sfc_l[np.nonzero(sfc_l)]
                    efc_l = np.array([list(itertools.chain.from_iterable(np.ndarray.tolist(efc)))])
                    efc_array = efc_l[np.nonzero(efc_l)]
                    res = np.corrcoef(sfc_array, efc_array)[1, 0]
                    corrResult.append(res)
                # visualization
                plt.figure(figsize=(20,5))
                plt.title(y+'_Goptimal_parameter_spacing')
                plt.plot(np.arange(0.01, 0.080, 0.001), corrResult, '*:b')
                plt.xticks(np.arange(0.01, 0.080, 0.002))
                save_fig = grp+'_'+y+'_G_optimal.png'
                plt.savefig(save_fig)
                df = df.append({'caseid':y, 'new_Go': tmp[int(np.argmax(corrResult))]}, ignore_index=True)
            except:
                continue
        # save csv file as output
        save_path = grp + '.csv'
        df.to_csv(save_path)

    # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
    # sns.heatmap(df1, ax=ax1, cmap='viridis')
    # subtit = '0306A_Go_' + str(i)
    # ax1.set_title(subtit)
    # sns.heatmap(df2, ax=ax2, cmap='viridis')
    # ax2.set_title('0306A_eFC')
    # figpth = '0306A_Go_'+str(i)+'.png'
    # plt.savefig(figpth)








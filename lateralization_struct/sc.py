
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
import io
from read_mat import Case
import seaborn as sns
import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/home/wayne/github/TVB_workflow/functions')
else:
    sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\functions')
from pathConverter import pathcon
from bootstrap import BootstrapTest
from permutation import PermutationTest, stats_calculator
from scipy import stats
import itertools

"""
Author: Yile
This script is designed to calculate the the direct weight connection

"""

# brain regions labels
regions = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMp-R','mTEMp-L','mTEMp-R']
cmap = sns.diverging_palette(230, 20, as_cmap=True)

def violin_plot(dataframe, column):
    fig = plt.figure(figsize=(10,10))
    plt.title('Homotopic-Streamlines')
    fig = sns.violinplot(x="grp", y=column, data=dataframe, capsize=.2,palette=["#66CDAA","#4682B4","#AB63FA","#FFA15A"])
    fig = sns.stripplot(x="grp", y=column, data=dataframe,color='black')
    fig = sns.pointplot(data=dataframe, x='grp', y=column, join=False, ci=None, color='red')
    # fig.set_ylim(-0.5, 1)
    # fig.set_yticks(np.arange(-0.5, 1, 0.1))
    plt.show()





if __name__ == '__main__':
    limbic = ['aCNG', 'mCNG', 'pCNG', 'HIP', 'PHG','AMY','sTEMp','mTEMp']
    grp_pools = ['SNC', 'NC','MCI', 'AD']
    start = time.time()
    pdList = []
    # fig, axs = plt.subplots(2, sharex = True, sharey = True, figsize=(12,8))
    # fig.suptitle("G frequency and Gamma")
    col = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
    head = ['grp', 'caseid']
    tmp_head = np.concatenate((head, limbic), axis=0)
    direct_sc_data = pd.DataFrame(columns=tmp_head)
    hetero_sc_data_left = pd.DataFrame(columns=tmp_head)
    hetero_sc_data_right = pd.DataFrame(columns=tmp_head)
    mean_group_matrix = np.zeros((16, 16, 4))
    for indx, grp in enumerate(grp_pools):
    # obtain the data path
        pth = pathcon('AUS/') + grp
        case_pools = os.listdir(pth)
        group_matrix = np.zeros((16, 16, len(case_pools)))
        for ind, caseid in enumerate(case_pools):
            ### the structural connectivity ###
            # SC lateralization
            # fig, (ax0, ax1) = plt.subplots(1,2, figsize=(15,5))
            # fig.suptitle(grp+'_'+caseid)
            datapath = pathcon('AUS/') + grp + '/' + caseid + '/weights.txt'
            scl = open(datapath,"r")
            lines = scl.read()
            tmp_d = pd.read_csv(io.StringIO(lines), sep='\t', header=None, index_col=None, engine="python")
            tmp_df = tmp_d.set_axis(regions, axis=0) # set the index
            df_sc = tmp_df.set_axis(regions, axis=1) # set the columns
            group_matrix[:,:,ind] = df_sc
            # sns.heatmap(df_sc,cmap='coolwarm')
            # plt.show()
            ### the lateralization ###

            #the direct lateralization
            # ind = np.arange(0, 16, 2)
            # m = []
            # for n in ind:
            #     tmp = df_sc.iloc[n+1, n]
            #     m.append(tmp)


            ###
            # The homotopic connectivity
            connect = np.array([])
            hetero = np.array([])
            tmp_hetero_left = np.array([])
            tmp_hetero_right = np.array([])
            for index in range(len(limbic)):
                # homotopic
                tmp_connect = df_sc.iloc[2*index+1, 2*index]
                connect = np.append(connect, tmp_connect)
            tmp_c = np.concatenate(([grp, caseid], connect)).reshape((10,1)).T
            tmp_c2 = pd.DataFrame(tmp_c, columns=tmp_head)
            direct_sc_data = direct_sc_data.append(tmp_c2, ignore_index=True)
            for index in range(len(regions)):
                # heterotopic
                tmp_left = []
                tmp_right = []
                if index % 2 == 0:
                    for indxx in range(1,17,2):
                        if indxx - index != 1:
                            tmp_left.append(df_sc.iloc[indxx, index])
                    tmp_hetero_left = np.append(tmp_hetero_left, sum(tmp_left)/7)
                else:
                    for indxx in range(0, 16, 2):
                        if index - indxx != 1:
                            tmp_right.append(df_sc.iloc[indxx+1, index])
                    tmp_hetero_right = np.append(tmp_hetero_right, sum(tmp_right)/7)
            tmp_left_c = np.concatenate(([grp, caseid], tmp_hetero_left)).reshape((10,1)).T
            tmp_left_c2 = pd.DataFrame(tmp_left_c, columns=tmp_head)
            tmp_right_c = np.concatenate(([grp, caseid], tmp_hetero_right)).reshape((10,1)).T
            tmp_right_c2 = pd.DataFrame(tmp_right_c, columns=tmp_head)
            hetero_sc_data_left = hetero_sc_data_left.append(tmp_left_c2, ignore_index=True)
            hetero_sc_data_right = hetero_sc_data_right.append(tmp_right_c2, ignore_index=True)
        end = time.time()
        logging.warning('Duration: {}'.format(end - start))
        mean_group_matrix[:,:,indx] = np.mean(group_matrix, axis=2)
    figure = plt.figure(figsize=(20,4))
    for i in range(len(grp_pools)):
        plt.subplot(1,4,i+1)
        plt.title(grp_pools[i])
        sns.heatmap(mean_group_matrix[:,:,i], cmap="coolwarm", vmin=0, vmax=80, xticklabels=regions, yticklabels=regions)
    plt.show()


    
    # avg_direct_sc = pd.DataFrame({'grp':direct_sc_data.iloc[:,0],'avg_direct_sc':np.mean(np.array(direct_sc_data.iloc[:, 2:].values).astype('float'), axis=1)})
    # print(stats_calculator(avg_direct_sc))
    # violin_plot(avg_direct_sc, "avg_direct_sc")

    # avg_hetero_sc_left = pd.DataFrame({'grp':hetero_sc_data_left.iloc[:,0],'avg_hetero_sc_left':np.mean(np.array(hetero_sc_data_left.iloc[:, 2:].values).astype('float'), axis=1)})

    # avg_hetero_sc_right = pd.DataFrame({'grp':hetero_sc_data_right.iloc[:,0],'avg_hetero_sc_right':np.mean(np.array(hetero_sc_data_right.iloc[:, 2:].values).astype('float'), axis=1)})
    # print(stats_calculator(avg_hetero_sc_right))
    # violin_plot(avg_hetero_sc_right, "avg_hetero_sc_right")

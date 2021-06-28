
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
import io
from read_mat import Case
import seaborn as sns

"""
Author: Yile
This script is designed to calculate the lateralization index

"""

# brain regions labels
regions = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R']
cmap = sns.diverging_palette(230, 20, as_cmap=True)


if __name__ == '__main__':
    grp_pools = ['AD','SNC','MCI', 'NC']
    start = time.time()
    pdList = []
    # fig, axs = plt.subplots(2, sharex = True, sharey = True, figsize=(12,8))
    # fig.suptitle("G frequency and Gamma")
    col = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
    fc_data = pd.DataFrame(columns=['grp','caseid','L&R_mean', 'L&R_var'])
    for grp in grp_pools:
        # obtain the data path
        pth = 'C:/Users/Wayne/tvb/LFP/'+grp
        case_pools = os.listdir(pth)
        for caseid in case_pools:
            ### the structural connectivity ###
            # SC lateralization
            fig, (ax0, ax1) = plt.subplots(1,2, figsize=(15,5))
            fig.suptitle(grp+'_'+caseid)
            datapath = 'C:/Users/Wayne/tvb/AUS/' + grp + '/' + caseid + '/weights.txt'
            scl = open(datapath,"r")
            lines = scl.read()
            tmp_d = pd.read_csv(io.StringIO(lines), sep='\t', header=None, index_col=None, engine="python")
            tmp_df = tmp_d.set_axis(regions, axis=0) # set the index
            df_sc = tmp_df.set_axis(regions, axis=1) # set the columns

            ### the lateralization ###

            #the direct lateralization
            ind = np.arange(0, 16, 2)
            m = []
            for n in ind:
                tmp = df_sc.iloc[n+1, n]
                m.append(tmp)
            #sc_data = sc_data.append({'grp':grp, 'caseid': caseid, 'L&R_mean': np.mean(m), 'L&R_var':np.var(m)}, ignore_index=True)

            # # the 


            # ### visualization ###
            # sns.heatmap(ax=ax0, data=df_sc, cmap=cmap)
            # ax0.set_title('Structural Connectivity')



            # ### the functional connectivity ###
            # # FC lateralization
            try:
                pth_efc = "C:/Users/Wayne/tvb/TS-4-Vik/"+grp+"-TS/"+ caseid +"/ROICorrelation_"+ caseid +".mat"
                tmp = Case(pth_efc)
                df_fc = pd.DataFrame.from_dict(tmp.readFile().get("ROICorrelation"))
                df_fc.columns = regions
                df_fc.index = regions
                # the direct lateralization
                ind = np.arange(0, 16, 2)
                m = []
                for n in ind:
                    tmp = df_sc.iloc[n+1, n]
                    m.append(tmp)
                fc_data = fc_data.append({'grp':grp, 'caseid': caseid, 'L&R_mean': np.mean(m), 'L&R_var':np.var(m)}, ignore_index=True)
            except FileNotFoundError:
                continue

                ### visualization ###
                sns.heatmap(ax=ax1, data=df_fc, cmap=cmap)
                ax1.set_title('Functional Connectivity')
                plt.show()
        end = time.time()
        logging.warning('Duration: {}'.format(end - start))
    # fc_data.to_csv('dirct_lat_fc.csv')
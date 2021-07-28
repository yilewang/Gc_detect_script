
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
sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\functions')
from bootstrap import BootstrapTest


"""
Author: Yile
This script is designed to calculate the the direct weight connection

"""

# brain regions labels
regions = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMp-R','mTEMp-L','mTEMp-R']
cmap = sns.diverging_palette(230, 20, as_cmap=True)


if __name__ == '__main__':
    limbic = ['aCNG', 'mCNG', 'pCNG', 'HIP', 'PHG','AMY','sTEMp','mTEMp']
    grp_pools = ['AD','SNC','MCI', 'NC']
    start = time.time()
    pdList = []
    # fig, axs = plt.subplots(2, sharex = True, sharey = True, figsize=(12,8))
    # fig.suptitle("G frequency and Gamma")
    col = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
    head = ['grp', 'caseid']
    tmp_head = np.concatenate((head, limbic), axis=0)
    direct_sc_data = pd.DataFrame(columns=tmp_head)
    for grp in grp_pools:
    # obtain the data path
        pth = 'C:/Users/Wayne/tvb/LFP/'+grp
        case_pools = os.listdir(pth)
        for caseid in case_pools:
            ### the structural connectivity ###
            # SC lateralization
            # fig, (ax0, ax1) = plt.subplots(1,2, figsize=(15,5))
            # fig.suptitle(grp+'_'+caseid)
            datapath = 'C:/Users/Wayne/tvb/AUS/' + grp + '/' + caseid + '/weights.txt'
            scl = open(datapath,"r")
            lines = scl.read()
            tmp_d = pd.read_csv(io.StringIO(lines), sep='\t', header=None, index_col=None, engine="python")
            tmp_df = tmp_d.set_axis(regions, axis=0) # set the index
            df_sc = tmp_df.set_axis(regions, axis=1) # set the columns
            ### the lateralization ###

            #the direct lateralization
            # ind = np.arange(0, 16, 2)
            # m = []
            # for n in ind:
            #     tmp = df_sc.iloc[n+1, n]
            #     m.append(tmp)
            connect = np.array([])
            for index, key in enumerate(limbic):
                tmp_connect = df_sc.iloc[2*index+1, 2*index]
                connect = np.append(connect, tmp_connect)
            tmp_c = np.concatenate(([grp, caseid], connect)).reshape((10,1)).T
            tmp_c2 = pd.DataFrame(tmp_c, columns=tmp_head)
            direct_sc_data = direct_sc_data.append(tmp_c2, ignore_index=True)



            # ### visualization ###
            # sns.heatmap(ax=ax0, data=df_sc, cmap=cmap)
            # ax0.set_title('Structural Connectivity')



            # ### the functional connectivity ###
            # # FC lateralization
            # try:
            #     pth_efc = "C:/Users/Wayne/tvb/TS-4-Vik/"+grp+"-TS/"+ caseid +"/ROICorrelation_"+ caseid +".mat"
            #     tmp = Case(pth_efc)
            #     df_fc = pd.DataFrame.from_dict(tmp.readFile().get("ROICorrelation"))
            #     df_fc.columns = regions
            #     df_fc.index = regions
            #     # the direct lateralization
            #     # ind = np.arange(0, 16, 2)
            #     # m = []
            #     # for n in ind:
            #     #     tmp = df_fc.iloc[n+1, n]
            #     #     m.append(tmp)
            #     # fc_data = fc_data.append({'grp':grp, 'caseid': caseid, 'L&R_mean': np.mean(m), 'L&R_var':np.var(m)}, ignore_index=True)
                
            # except FileNotFoundError:
            #     continue

                # ### visualization ###
                # sns.heatmap(ax=ax1, data=df_fc, cmap=cmap)
                # ax1.set_title('Functional Connectivity')
                # plt.show()
        end = time.time()
        logging.warning('Duration: {}'.format(end - start))
    

    for key in limbic:
        plt.figure(figsize=(15, 5))
        AD_sample = np.hstack(direct_sc_data.loc[direct_sc_data['grp'].isin(['AD']), [key]].values).astype('float')
        MCI_sample = np.hstack(direct_sc_data.loc[direct_sc_data['grp'].isin(['MCI']), [key]].values).astype('float')
        NC_sample = np.hstack(direct_sc_data.loc[direct_sc_data['grp'].isin(['NC']), [key]].values).astype('float')
        SNC_sample = np.hstack(direct_sc_data.loc[direct_sc_data['grp'].isin(['SNC']), [key]].values).astype('float')

        ad_CI, ad_dis = BootstrapTest(AD_sample, 1000)
        mci_CI, mci_dis = BootstrapTest(MCI_sample, 1000)
        nc_CI, nc_dis = BootstrapTest(NC_sample, 1000)
        snc_CI, snc_dis = BootstrapTest(SNC_sample, 1000)

        
        plt.title(key)
        plt.hist(snc_dis, bins='auto', color= "#FFA15A", label='SNC', alpha = 0.5)
        # plt.axvline(x=np.round(snc_CI[0],3), label='CI at {}'.format(np.round(snc_CI,3)),c="#FFA15A", linestyle = 'dashed')
        # plt.axvline(x=np.round(snc_CI[1],3),  c="#FFA15A", linestyle = 'dashed')
        plt.hist(nc_dis, bins='auto', color="#AB63FA", label='NC', alpha = 0.5)
        # plt.axvline(x=np.round(nc_CI[0],3), label='CI at {}'.format(np.round(nc_CI,3)),c="#AB63FA", linestyle = 'dashed')
        # plt.axvline(x=np.round(nc_CI[1],3),  c="#AB63FA", linestyle = 'dashed')
        plt.hist(mci_dis, bins='auto', color="#4682B4", label='MCI', alpha = 0.5)
        # plt.axvline(x=np.round(mci_CI[0],3), label='CI at {}'.format(np.round(mci_CI,3)),c="#4682B4", linestyle = 'dashed')
        # plt.axvline(x=np.round(mci_CI[1],3),  c="#4682B4", linestyle = 'dashed')
        plt.hist(ad_dis, bins='auto', color="#66CDAA", label='AD', alpha = 0.5)
        # plt.axvline(x=np.round(ad_CI[0],3), label='CI at {}'.format(np.round(ad_CI,3)),c="#66CDAA", linestyle = 'dashed')
        # plt.axvline(x=np.round(ad_CI[1],3),  c="#66CDAA", linestyle = 'dashed')
        plt.legend()
        plt.show()    




    # ind_key = key+'_lat_sc.csv'
    # sc_data.to_csv(ind_key)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\functions')
from bootstrap import BootstrapTest
from permutation import PermutationTest
import pandas as pd
from scipy.stats import mannwhitneyu
import os


def add_star(data):
    if data <= 0.001:
        tmp_data = str(data) + '***'
    elif data <= 0.01:
        tmp_data = str(data) + '**'
    elif data < 0.05:
        tmp_data = str(data) + '*'
    else:
        tmp_data = data
    return tmp_data
    

# import data

# G values
G_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx', sheet_name='Gc_Go')

# Integration
Integration_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Ignition_regroups.xlsx', sheet_name='Integration')

# Ignition
Ignition_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx', sheet_name='Ignition')
Ig_wholebrain_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Ignition_regroups.xlsx', sheet_name='Ignition_whole_brain')
ROI_Ig_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Ignition_regroups.xlsx', sheet_name='ROI-Ignition Group')

# The Freq_Amp_Delay
Mix_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/mix_final.xlsx')

# the key
G_datakey = ['Gmax-Gc','Go','Gc','Go-Gc','Gmax']
Ignition_key = ['aCNG', 'mCNG', 'pCNG', 'HIP', 'PHG', 'AMY', 'sTEMp', 'mTEMp']
Ig_wholebrain_key = ['Ignition_whole_brain']
ROI_key = ['Ignition']
Inte_key = ['Integration']
Mix_key = ['freqL_Gamma',	'freqR_Gamma',	'freqL_Theta',	'freqR_Theta', 'ampL_gamma_ratio', 'ampR_gamma_ratio', 'ampL_theta_ratio','ampR_theta_ratio', 'LI_gamma_ratio','LI_theta_ratio','ampL_abs','ampR_abs','LI_abs', 'Delay']


sheet_columns = ['type','features','SNC_AD','SNC_MCI', 'SNC_NC', 'AD_MCI','AD_NC','MCI_NC']

MixData = pd.DataFrame(columns=sheet_columns) 
# SN_MAData = pd.DataFrame(columns=['type', 'features', 'SN_MA'])
# SN_other = pd.DataFrame(columns=['type', 'features', 'SN_MCI', 'SN_AD'])
regroups = 'groups'


def stats_generator(pdd, key, type: str, dataset, graphic=True):
    types = ['permutation', 'mann']
    if type not in types:
        raise ValueError("Invalid type. Expected one of: %s" % types)

    
    AD_sample = np.hstack(pdd.loc[pdd[regroups].isin(['4.AD']), [key]].values)
    MCI_sample = np.hstack(pdd.loc[pdd[regroups].isin(['3.aMCI']), [key]].values)
    NC_sample = np.hstack(pdd.loc[pdd[regroups].isin(['2.NC']), [key]].values)
    SNC_sample = np.hstack(pdd.loc[pdd[regroups].isin(['1.SNC']), [key]].values)

    # SNC_NC_sample = np.hstack(pdd.loc[pdd[regroups].isin(['1.SNC+NC']), [key]].values)
    # MCI_AD_sample = np.hstack(pdd.loc[pdd[regroups].isin(['2.MCI+AD']), [key]].values)
    # MCI_sample = np.hstack(pdd.loc[pdd[regroups].isin(['2.MCI']), [key]].values)
    # AD_sample = np.hstack(pdd.loc[pdd[regroups].isin(['3.AD']), [key]].values)


    # ad_CI, ad_dis = BootstrapTest(AD_sample, 1000)
    # mci_CI, mci_dis = BootstrapTest(MCI_sample, 1000)
    # nc_CI, nc_dis = BootstrapTest(NC_sample, 1000)
    # snc_CI, snc_dis = BootstrapTest(SNC_sample, 1000)
    # plt.figure(figsize=(15, 5))
    # plt.title(f"Bootstrap of {key}")

    # plt.hist(snc_dis, bins='auto', color= "#66CDAA", label='SNC', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(snc_CI[0],3), label='CI at {}'.format(np.round(snc_CI,3)),c="#FFA15A", linestyle = 'dashed')
    # # plt.axvline(x=np.round(snc_CI[1],3),  c="#FFA15A", linestyle = 'dashed')
    # plt.hist(nc_dis, bins='auto', color="#4682B4", label='NC', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(nc_CI[0],3), label='CI at {}'.format(np.round(nc_CI,3)),c="#AB63FA", linestyle = 'dashed')
    # # plt.axvline(x=np.round(nc_CI[1],3),  c="#AB63FA", linestyle = 'dashed')
    # plt.hist(mci_dis, bins='auto', color="#AB63FA", label='MCI', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(mci_CI[0],3), label='CI at {}'.format(np.round(mci_CI,3)),c="#4682B4", linestyle = 'dashed')
    # # plt.axvline(x=np.round(mci_CI[1],3),  c="#4682B4", linestyle = 'dashed')
    # plt.hist(ad_dis, bins='auto', color="#FFA15A", label='AD', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(ad_CI[0],3), label='CI at {}'.format(np.round(ad_CI,3)),c="#66CDAA", linestyle = 'dashed')
    # # plt.axvline(x=np.round(ad_CI[1],3),  c="#66CDAA", linestyle = 'dashed')
    # plt.legend()
    # my_path = os.path.abspath('C:/Users/Wayne/tvb/stat_data/bootstrap_results') # Figures out the absolute path for you in case your working directory moves around.
    # my_file = f"{key}.png"
    # plt.savefig(os.path.join(my_path, my_file))



    if type == 'mann':
        _, snc_ad = mannwhitneyu(SNC_sample, AD_sample)
        _, nc_ad = mannwhitneyu(NC_sample, AD_sample)
        _, mci_ad = mannwhitneyu(MCI_sample, AD_sample)
        _, snc_nc = mannwhitneyu(SNC_sample, NC_sample)
        _, snc_mci = mannwhitneyu(SNC_sample, MCI_sample)
        _, nc_mci = mannwhitneyu(NC_sample, MCI_sample)


        # _, sn_mci = mannwhitneyu(SNC_NC_sample, MCI_sample)
        # _, sn_ad = mannwhitneyu(SNC_NC_sample, AD_sample)

    if type == 'permutation':
        graphic = False

        snc_ad = PermutationTest(SNC_sample, AD_sample,10000, graphic)
        nc_ad = PermutationTest(NC_sample, AD_sample, 10000, graphic)
        mci_ad = PermutationTest(MCI_sample, AD_sample, 10000, graphic)
        snc_nc = PermutationTest(SNC_sample, NC_sample, 10000, graphic)
        snc_mci = PermutationTest(SNC_sample, MCI_sample, 10000, graphic)
        nc_mci = PermutationTest(NC_sample, MCI_sample, 10000, graphic)

        # sn_mci = PermutationTest(SNC_NC_sample, MCI_sample, 10000, graphic)
        # sn_ad = PermutationTest(SNC_NC_sample, AD_sample, 10000, graphic)

    snc_ad = add_star(snc_ad)
    nc_ad = add_star(nc_ad)
    mci_ad = add_star(mci_ad)
    snc_nc = add_star(snc_nc)
    snc_mci = add_star(snc_mci)
    nc_mci = add_star(nc_mci)

    # sn_mci = add_star(sn_mci)
    # sn_ad = add_star(sn_ad)



    dataset = dataset.append({'type':type, 'features':key,'SNC_AD':snc_ad,'SNC_MCI':snc_mci, 'SNC_NC':snc_nc, 'AD_MCI':mci_ad,'AD_NC':nc_ad,'MCI_NC':nc_mci} ,ignore_index=True)

    #dataset = dataset.append({'type':type, 'features':key,'SN_MA':all_groups_combine}, ignore_index=True)
    # dataset = dataset.append({'type':type, 'features':key,'SN_MCI':sn_mci, 'SN_AD':sn_ad}, ignore_index=True)

    return dataset

for key in G_datakey:
    MixData = stats_generator(G_table, key, 'mann', MixData)
    MixData = stats_generator(G_table, key, 'permutation', MixData)

for key in Ignition_key:
    MixData = stats_generator(Ignition_table, key, 'mann', MixData)
    MixData = stats_generator(Ignition_table, key, 'permutation', MixData)


MixData = stats_generator(Integration_table, 'Integration', 'mann', MixData)
MixData = stats_generator(Integration_table, 'Integration', 'permutation', MixData)

MixData = stats_generator(Ig_wholebrain_table, 'Ignition_whole_brain', 'mann', MixData)
MixData = stats_generator(Ig_wholebrain_table, 'Ignition_whole_brain', 'permutation', MixData)

MixData = stats_generator(ROI_Ig_table, 'Ignition', 'mann', MixData)
MixData = stats_generator(ROI_Ig_table, 'Ignition', 'permutation', MixData)

for key in Mix_key:
    MixData = stats_generator(Mix_table, key, 'mann', MixData)
    MixData = stats_generator(Mix_table, key, 'permutation', MixData)

MixData.to_excel('final_stat_table.xlsx')










# for key in G_datakey:
#     SN_MAData = stats_generator(G_table, key, 'mann', SN_MAData)
#     SN_MAData = stats_generator(G_table, key, 'permutation', SN_MAData)

# for key in Ignition_key:
#     SN_MAData = stats_generator(Ignition_table, key, 'mann', SN_MAData)
#     SN_MAData = stats_generator(Ignition_table, key, 'permutation', SN_MAData)


# SN_MAData = stats_generator(Integration_table, 'Integration', 'mann', SN_MAData)
# SN_MAData = stats_generator(Integration_table, 'Integration', 'permutation', SN_MAData)

# SN_MAData = stats_generator(Ig_wholebrain_table, 'Ignition_whole_brain', 'mann', SN_MAData)
# SN_MAData = stats_generator(Ig_wholebrain_table, 'Ignition_whole_brain', 'permutation', SN_MAData)

# SN_other = stats_generator(Ig_wholebrain_table, 'Ignition_whole_brain', 'mann', SN_other)
# SN_other = stats_generator(Ig_wholebrain_table, 'Ignition_whole_brain', 'permutation', SN_other)


# SN_MAData = stats_generator(ROI_Ig_table, 'Ignition', 'mann', SN_MAData)
# SN_MAData = stats_generator(ROI_Ig_table, 'Ignition', 'permutation', SN_MAData)




# for key in Mix_key:
#     SN_MAData = stats_generator(Mix_table, key, 'mann', SN_MAData)
#     SN_MAData = stats_generator(Mix_table, key, 'permutation', SN_MAData)

# SN_other.to_excel('ig_whole_sn.xlsx')




    # ad_CI, ad_dis = BootstrapTest(AD_sample, 1000)
    # mci_CI, mci_dis = BootstrapTest(MCI_sample, 1000)
    # nc_CI, nc_dis = BootstrapTest(NC_sample, 1000)
    # snc_CI, snc_dis = BootstrapTest(SNC_sample, 1000)
    # plt.figure(figsize=(15, 5))
    # plt.title(f"Bootstrap of {key}")

    # plt.hist(snc_dis, bins='auto', color= "#66CDAA", label='SNC', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(snc_CI[0],3), label='CI at {}'.format(np.round(snc_CI,3)),c="#FFA15A", linestyle = 'dashed')
    # # plt.axvline(x=np.round(snc_CI[1],3),  c="#FFA15A", linestyle = 'dashed')
    # plt.hist(nc_dis, bins='auto', color="#4682B4", label='NC', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(nc_CI[0],3), label='CI at {}'.format(np.round(nc_CI,3)),c="#AB63FA", linestyle = 'dashed')
    # # plt.axvline(x=np.round(nc_CI[1],3),  c="#AB63FA", linestyle = 'dashed')
    # plt.hist(mci_dis, bins='auto', color="#AB63FA", label='MCI', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(mci_CI[0],3), label='CI at {}'.format(np.round(mci_CI,3)),c="#4682B4", linestyle = 'dashed')
    # # plt.axvline(x=np.round(mci_CI[1],3),  c="#4682B4", linestyle = 'dashed')
    # plt.hist(ad_dis, bins='auto', color="#FFA15A", label='AD', alpha = 0.5,histtype='bar', ec='black')
    # # plt.axvline(x=np.round(ad_CI[0],3), label='CI at {}'.format(np.round(ad_CI,3)),c="#66CDAA", linestyle = 'dashed')
    # # plt.axvline(x=np.round(ad_CI[1],3),  c="#66CDAA", linestyle = 'dashed')
    # plt.legend()
    # plt.show()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\functions')
from bootstrap import BootstrapTest
from permutation import PermutationTest
import pandas as pd



coData = pd.read_excel('C:/Users/Wayne/tvb/TVB_workflow/new_g_oscillation/Gc_Go.xlsx', index_col=0)

key = 'Gc'

AD_sample = np.hstack(coData.loc[coData['groups'].isin(['4.AD']), [key]].values)
MCI_sample = np.hstack(coData.loc[coData['groups'].isin(['3.aMCI']), [key]].values)
NC_sample = np.hstack(coData.loc[coData['groups'].isin(['2.NC']), [key]].values)
SNC_sample = np.hstack(coData.loc[coData['groups'].isin(['1.SNC']), [key]].values)

PermutationTest(SNC_sample, AD_sample,5000, True)

# ad_CI, ad_dis = BootstrapTest(AD_sample, 1000)
# mci_CI, mci_dis = BootstrapTest(MCI_sample, 1000)
# nc_CI, nc_dis = BootstrapTest(NC_sample, 1000)
# snc_CI, snc_dis = BootstrapTest(SNC_sample, 1000)


# plt.figure(figsize=(15, 5))
# plt.title('Bootstrap of four groups')

# sns.histplot(data=coData, x='Gc', hue='groups')
# plt.show()



# plt.hist(snc_dis, bins='auto', color= "#FFA15A", label='SNC', alpha = 0.5)
# # plt.axvline(x=np.round(snc_CI[0],3), label='CI at {}'.format(np.round(snc_CI,3)),c="#FFA15A", linestyle = 'dashed')
# # plt.axvline(x=np.round(snc_CI[1],3),  c="#FFA15A", linestyle = 'dashed')
# plt.hist(nc_dis, bins='auto', color="#AB63FA", label='NC', alpha = 0.5)
# # plt.axvline(x=np.round(nc_CI[0],3), label='CI at {}'.format(np.round(nc_CI,3)),c="#AB63FA", linestyle = 'dashed')
# # plt.axvline(x=np.round(nc_CI[1],3),  c="#AB63FA", linestyle = 'dashed')
# plt.hist(mci_dis, bins='auto', color="#4682B4", label='MCI', alpha = 0.5)
# # plt.axvline(x=np.round(mci_CI[0],3), label='CI at {}'.format(np.round(mci_CI,3)),c="#4682B4", linestyle = 'dashed')
# # plt.axvline(x=np.round(mci_CI[1],3),  c="#4682B4", linestyle = 'dashed')
# plt.hist(ad_dis, bins='auto', color="#66CDAA", label='AD', alpha = 0.5)
# # plt.axvline(x=np.round(ad_CI[0],3), label='CI at {}'.format(np.round(ad_CI,3)),c="#66CDAA", linestyle = 'dashed')
# # plt.axvline(x=np.round(ad_CI[1],3),  c="#66CDAA", linestyle = 'dashed')
# plt.legend()
# plt.show()

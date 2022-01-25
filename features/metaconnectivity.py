# #!/usr/bin/python
# import pandas as pd
# import numpy as np
# import sys
# sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\new_g_optimal')
# sys.path.append('C:\\Users\\Wayne\\tvb\\Network-science-Toolbox\\Python')
# # sys.path.append('/home/wayne/github/TVB_workflow/new_g_optimal')
# # sys.path.append('/home/wayne/github/Network-science-Toolbox/Python')
# from TS2dFCstream import TS2dFCstream
# from dFCstream2Trimers import dFCstream2Trimers
# from dFCstream2MC import dFCstream2MC
# from read_mat import Case
# from read_corrMatrix import ReadRaw
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats.stats import pearsonr
# import itertools
# import os

# """
# @Author: Yile Wang

# This script is used to calculate the homotopic meta-connectivity in four groups, SNC, NC, MCI, AD
# """

# # brain region labels for your reference
# regions = np.array(['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R'])
# regionsHalf = np.array(['aCNG', 'mCNG','pCNG','HIP','PHG','AMY','sTEMp', 'mTEMp'])
# groups = np.array(['SNC', 'NC', 'MCI','AD'])


# if __name__ == "__main__":
#     HomotopicMC = pd.DataFrame(columns=['grp','caseid', 'Homo','aCNG','mCNG','pCNG','HIP','PHG','AMY','sTEMp','mTEMp' ])
#     HetertopicMC  = pd.DataFrame(columns=['grp','caseid', 'Hetero','aCNG','mCNG','pCNG','HIP','PHG','AMY','sTEMp','mTEMp' ])

#     regions_num = range(len(regions))
#     iterName = list(itertools.combinations(regions, 2))
#     for a, x in enumerate(iterName):
#         tmp = ' & '.join(x)
#         iterName[a] = tmp




#     for grp in groups:
#         # subject case ids
#         ldir = os.listdir("C:/Users/Wayne/tvb/TS-4-Vik/"+ grp+'-TS')
#         # ldir = os.listdir('/home/wayne/TS-4-Vik/'+grp+'-TS/')
#         for y in ldir:
#             # import empirical functional connectivity
#             # Here is the path of the mat file of the FC data
#             pth_efc = "C:/Users/Wayne/tvb/TS-4-Vik/"+ grp+'-TS/'+ y +"/ROISignals_"+ y +".mat"
#             # pth_efc = "/home/wayne/TS-4-Vik/"+grp+"-TS/"+ y +"/ROISignals_"+ y +".mat"
#             a2 = Case(pth_efc)
#             df2 = pd.DataFrame.from_dict(a2.readFile().get("ROISignals"))
#             df2.columns = regions
#             # calculate the meta-connectivity, using existing script:
#             dFCstream = TS2dFCstream(df2.to_numpy(), 5, None, '2D')
#             MC_MC = dFCstream2MC(dFCstream)
#             MC_df = pd.DataFrame(MC_MC, index=iterName, columns=iterName)

#             # create dataframe of MetaConnectivtiy
#             keysBase = []
#             keysNum = []
#             for a, x in enumerate(regions):
#                 # the list without a (target region)
#                 newList = list(range(16))
#                 newList.remove(a)
#                 keysNum.append(newList)
#                 keysBan = []
#                 for c in newList:
#                     if a < c:
#                         tmpKey = x + ' & ' + regions[c]
#                         keysBan.append(tmpKey)
#                     else:
#                         tmpKey = regions[c] + ' & ' + x
#                         keysBan.append(tmpKey)
#                 keysBase.append(keysBan)
            
#             # Homotopic Connectivity
#             for i in range(0,15,2):
#                 j = i+1
#                 homotopic = MC_df.loc[keysBase[i,j]]
#         break
                    
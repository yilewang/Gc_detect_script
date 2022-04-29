#！/bin/python

from mne_connectivity.viz import circular_layout, plot_connectivity_circle
import numpy as np
import pandas as pd
import io
import os
import matplotlib.pyplot as plt

N = 8
node_names = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMp-R','mTEMp-L','mTEMp-R']
split_codes = ['aCNG-L', 'mCNG-L', 'pCNG-L', 'HIP-L', 'PHG-L', 'AMY-L', 'sTEMp-L','mTEMp-L', 'aCNG-R', 'mCNG-R', 'pCNG-R', 'HIP-R', 'PHG-R', 'AMY-R', 'sTEMp-R','mTEMp-R']


def find_third(x: list):
    num = int(len(x)/2)
    n1, n2, n3 = num+np.int(num/3)-1, num+np.int(num/3*2)-1, num+np.int(num-1)
    return x[n1], x[n2], x[n3]

grp_pools = ['SNC', 'NC','MCI', 'AD']
mean_group_matrix = np.zeros((16, 16, 4))
fig1 = plt.figure(figsize=(3,3))
for indx, grp in enumerate(grp_pools):
# obtain the data path
    pth = 'C:/Users/Wayne/tvb/AUS/' + grp
    case_pools = os.listdir(pth)
    group_matrix = np.zeros((16, 16, len(case_pools)))
    for ind, caseid in enumerate(case_pools):
        ### the structural connectivity ###
        # SC lateralization
        # fig, (ax0, ax1) = plt.subplots(1,2, figsize=(15,5))
        # fig.suptitle(grp+'_'+caseid)
        datapath = 'C:/Users/Wayne/tvb/AUS/' + grp + '/' + caseid + '/weights.txt'
        scl = open(datapath,"r")
        lines = scl.read()
        tmp_d = pd.read_csv(io.StringIO(lines), sep='\t', header=None, index_col=None, engine="python")
        tmp_df = tmp_d.set_axis(node_names, axis=0) # set the index
        df_sc = tmp_df.set_axis(node_names, axis=1) # set the columns
        df_sc = df_sc.reindex(index=split_codes, columns=split_codes)
        group_matrix[:,:,ind] = df_sc

        x = np.sort(np.ndarray.flatten(np.tril(group_matrix[:,:,0])))
        print(find_third(x))

    node_order = list()
    node_order.extend(split_codes[:8])  # reverse the order
    node_order.extend(split_codes[8:][::-1])
    node_angles = circular_layout(split_codes, node_order, start_pos=90,
                                group_boundaries=[0, len(split_codes) / 2])
    # node_angles[0:4] += 15
    # node_angles[4:8] -= 15
    # node_angles[8:12] -= 15
    # node_angles[12:16] += 15

    fig, axis = plot_connectivity_circle(np.tril(np.array(np.mean(group_matrix, 2))), split_codes, node_angles=node_angles,colormap='Greens', facecolor='white', textcolor='black', colorbar=False, linewidth=3, title=grp, fontsize_names=10, show=False, fig = fig1, vmin=0, vmax=132.0)
    #fig.savefig(f"{grp}_sc.png", dpi=300)
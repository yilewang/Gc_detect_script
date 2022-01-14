#！/bin/python

from mne_connectivity.viz import circular_layout, plot_connectivity_circle
import numpy as np
import pandas as pd
import io

N = 8
node_names = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMp-R','mTEMp-L','mTEMp-R']
split_codes = ['aCNG-L', 'mCNG-L', 'pCNG-L', 'HIP-L', 'PHG-L', 'AMY-L', 'sTEMp-L','mTEMp-L', 'aCNG-R', 'mCNG-R', 'pCNG-R', 'HIP-R', 'PHG-R', 'AMY-R', 'sTEMp-R','mTEMp-R']

datapath = r'C:\Users\Wayne\tvb\AUS\SNC\8709A\weights.txt'
scl = open(datapath,"r")
lines = scl.read()
tmp_d = pd.read_csv(io.StringIO(lines), sep='\t', header=None, index_col=None, engine="python")
tmp_df = tmp_d.set_axis(node_names, axis=0) # set the index
df_sc = tmp_df.set_axis(node_names, axis=1) # set the columns

df_sc = df_sc.reindex(index=split_codes, columns=split_codes)

node_order = list()
node_order.extend(split_codes[:8])  # reverse the order
node_order.extend(split_codes[8:][::-1])
node_angles = circular_layout(split_codes, node_order, start_pos=90,
                              group_boundaries=[0, len(split_codes) / 2])

fig, axis = plot_connectivity_circle(np.array(df_sc), split_codes, node_angles=node_angles,colormap='Blues', facecolor='white', textcolor='black', colorbar=False, linewidth=10, title="Structural Connectivity in Limbic Subnetwork", fontsize_names=10)
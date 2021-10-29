from math import perm
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
import itertools
import scipy


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





def stats_calculator(datatable):
    """
    Args:
        datatable, including grouping variable
    Output
        the permutation resutls for each groups
    """
    groups = pd.unique(datatable.loc[:,'grp'])
    groups_num = range(len(groups))

    comba = list(itertools.combinations(groups_num, 2))

    # give labels to comba
    comba_with_name = []
    for a, x in enumerate(comba):
        tmp_one = (groups[x[0]], groups[x[1]])
        comba_with_name.append(tmp_one)

    overall_permu =np.zeros((len(datatable.columns), len(comba)))
    for a,col in enumerate(datatable.columns):
        if isinstance(datatable.iloc[0,a], (np.integer, np.float64)):
            deList = [[] for i in groups_num]
            for x in groups_num:
                deList[x] = datatable.loc[datatable['grp'].isin([groups[x]]), [col]].values.flatten()
            tmp_list = np.array([])
            for y in comba:
                p_value = PermutationTest(deList[y[0]], deList[y[1]], iteration = 10000, visualization = False)
                tmp_list = np.append(tmp_list, p_value)
            overall_permu[a, :] = tmp_list

        else:
            continue
    dataframe = pd.DataFrame(overall_permu, index=datatable.columns, columns=comba_with_name)
    # drop the 0 value columns in original datatable
    for col in datatable.columns:
        if np.all(dataframe.loc[col, :].values):
            continue
        else:
            dataframe.drop([col], axis=0, inplace = True)
    
    # add asterisk sign
    for i in range(np.shape(dataframe)[0]):
        for k in range(np.shape(dataframe)[1]):
            dataframe.iloc[i,k] = add_star(dataframe.iloc[i,k])
    return dataframe


def bootstrap_groups(datatable, iternation:int, col:str):
    
    groups = pd.unique(datatable.loc[:,'groups'])
    palette = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
    #palette = sns.color_palette(None, len(groups))
    container = np.zeros((len(groups), iternation))
    for num in range(len(groups)):
        rmd = datatable.loc[datatable['groups'].isin([groups[num]]), col].values.flatten()
        container[num] = BootstrapTest(rmd, iternation)[1]
    fig=plt.figure(figsize=(15,5))
    for num in range(len(groups)):
        sns.histplot(x=container[num,:], color=palette[num], alpha=0.7, label=groups[num])
        plt.legend()
    plt.show()


G_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx', sheet_name='Gc_Go')
# Mix_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/mix_final.xlsx')
# amp = pd.read_excel('C:/Users/Wayne/tvb/amp_abs.xlsx')
# freq = pd.read_excel('C:/Users/Wayne/tvb/freq.xlsx')
ignition=pd.read_excel('C:/Users/Wayne/R.TVB_Ignition/ignition_table_merge.xlsx', sheet_name='ignition_merge')
# ignition=pd.read_excel('C:/Users/Wayne/tvb/stat_data/Ignition_regroups.xlsx', sheet_name='Ignition_whole_brain')
# integration = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Ignition_regroups.xlsx', sheet_name='Integration')
# freq_amp = pd.read_excel('C:/Users/Wayne/R.TVB_Ignition/freq_amp_combination.xlsx')


stats_calculator(ignition).to_excel('ignition_single_regions.xlsx')

# bootstrap_groups(G_table, iternation=10000, col='Gc')

# levene test
# print(scipy.stats.levene(G_table.loc[G_table['groups'].isin(['4.AD']), 'Gc'].values.flatten(), G_table.loc[G_table['groups'].isin(['2.NC']), 'Go'].values.flatten(), center='mean'))

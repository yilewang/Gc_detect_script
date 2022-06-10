#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
import sys
sys.path.append('C:\\Users\\Wayne\\tvb\\TVB_workflow\\functions')
from permutation import PermutationTest
"""
This is a bootstrap test python script
Author: Yile Wang
Date: 07/27/2022

"""


def BootstrapTest(x,iteration, visualization = False):
    """
    Args:   
        x: data list1 1-d array
        iteration: iteration number for the test
        visualization (boolean): the default value is False. If it is True, the bootstrap histgram will be generated
    Returns:
        CI the bootstrap test
        bootstrap distribution

    """
    # transfer data to array format
    x = np.array(x)
    box = np.array([])
    i = 0
    while i < iteration:
        idx_x = np.random.choice(x, size=x.shape[0], replace=True)
        p_mean = np.mean(idx_x)
        box = np.append(box, p_mean)
        i+=1
    bt_mean = np.mean(box)
    sorted_box = np.sort(box)
    low_CI = np.percentile(sorted_box, 2.5)
    high_CI = np.percentile(sorted_box, 97.5)
    CI = (low_CI, high_CI)
    
    #p_value = (box[box > np.mean(x)].shape[0] + 1) / (iteration + 1) # correction
    print(f"The CI of the Bootstrap Test is: {CI}")
    
    # visualization
    if visualization == True:
        plt.figure(figsize=(9,8))
        sns.histplot(data=box, bins='auto')
        plt.axvline(x=np.round(CI[0],3), label='2.5% CI at {}'.format(np.round(CI[0],3)),c='g', linestyle = 'dashed')
        plt.axvline(x=np.round(CI[1],3), label='97.5% CI at {}'.format(np.round(CI[1],3)), c='g', linestyle = 'dashed')
        plt.axvline(x = np.mean(x), c='r', label = 'original mean at {}'.format(np.mean(x)))
        plt.axvline(x = np.round(bt_mean, 3), c='r', label = 'bootstrap mean at {}'.format(np.round(bt_mean, 3)), linestyle='dashed')
        plt.legend()
        plt.show()
    return CI, box
    

    
# ############################
# ## Test codes:
# x = np.random.random_sample((100,))
# BootstrapTest(x, 1000, True)
# ############################


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




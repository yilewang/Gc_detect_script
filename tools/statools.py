#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import itertools
import pandas as pd

"""
This is a python script with statistical tools
Author: Yile Wang
Date: 06/10/2022
"""

def bootstrap_test(x,iteration, visual = False):
    """
    A script for bootstrap analysis
    Parameters:
    ---------------------   
        x: data list1 1-d array
        iteration: iteration number for the test
        visual (boolean): the default value is False. If it is True, the bootstrap histgram will be generated
    Returns:
    ---------------------
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
    
    # visual
    if visual == True:
        plt.figure(figsize=(9,8))
        sns.histplot(data=box, bins='auto')
        plt.axvline(x=np.round(CI[0],3), label='2.5% CI at {}'.format(np.round(CI[0],3)),c='g', linestyle = 'dashed')
        plt.axvline(x=np.round(CI[1],3), label='97.5% CI at {}'.format(np.round(CI[1],3)), c='g', linestyle = 'dashed')
        plt.axvline(x = np.mean(x), c='r', label = 'original mean at {}'.format(np.mean(x)))
        plt.axvline(x = np.round(bt_mean, 3), c='r', label = 'bootstrap mean at {}'.format(np.round(bt_mean, 3)), linestyle='dashed')
        plt.legend()
        plt.show()
    return CI, box



def permutation_test(x,y,iteration, visual = False):
    """
    Args:   
        x: data list1 1-d array
        y: data list2 1-d array
        iteration: iteration number for the test
        visual (boolean): the default value is False. If it is True, the permutation histgram will be generated
    Returns:
        p-value of the permutation test

    """
    # transfer data to array format
    if len(y) > len(x):
        tmp_x = y
        tmp_y = x
    else:
        tmp_x = x
        tmp_y = y
    x = np.array(tmp_x)
    y = np.array(tmp_y)
    np.hstack((x,y))
    orig_mean = abs(np.mean(x) - np.mean(y))
    Z = np.hstack((x, y))
    box = np.array([])
    i = 0
    while i < iteration:
        idx_x = np.random.choice(Z, size= x.shape[0], replace=True)
        idx_y = np.asarray([ele for ele in Z if ele not in idx_x])
        p_mean = np.mean(idx_x) - np.mean(idx_y)
        box = np.append(box, p_mean)
        i+=1
    permu_mean = np.mean(box)
    p_value = (box[box > orig_mean].shape[0] + 1) / (iteration + 1) # correction

    
    # visual
    if visual == True:
        print(f"The P-value of the Permutation Test is: {p_value}")
        plt.figure(figsize=(9,8))
        sns.histplot(data=box, bins='auto')
        plt.axvline(x=np.round(permu_mean,3), label='Permutation Mean at {}'.format(np.round(permu_mean,3)),c='g')
        plt.axvline(x=orig_mean, label='Original Mean at {}'.format(orig_mean), c='r', linestyle = 'dashed')
        plt.legend()
        plt.show()
    return p_value

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
    groups = pd.unique(datatable.loc[:,'group'])
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
                deList[x] = datatable.loc[datatable['group'].isin([groups[x]]), [col]].values.flatten()
            tmp_list = np.array([])
            for y in comba:
                p_value = permutation_test(deList[y[0]], deList[y[1]], iteration = 10000, visual = False)
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
        container[num] = bootstrap_test(rmd, iternation)[1]
    fig=plt.figure(figsize=(15,5))
    for num in range(len(groups)):
        sns.histplot(x=container[num,:], color=palette[num], alpha=0.7, label=groups[num])
        plt.legend()
    plt.show()



#############################
### Test codes:
# x = np.random.random_sample((20,))
# y = np.random.random_sample((12,))
# x = [1,2,3,4,5]
# y = [6,7]
# xy = x+y
# print(permutation_test(x, y, 10000, False))
# data_all = pd.read_excel('/mnt/c/Users/Wayne/tvb/amp_pro_final.xlsx', sheet_name='amp_pro_final')
# stats_calculator(data_all).to_excel('data_amp_pro.xlsx')
#############################



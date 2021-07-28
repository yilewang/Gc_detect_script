#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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



#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
"""
This is a permutation test python script
Author: Yile Wang
Date: 07/11/2021

"""


def PermutationTest(x,y,iteration, visualization = False):
    """
    Args:   
        x: data list1 1-d array
        y: data list2 1-d array
        iteration: iteration number for the test
        visualization (boolean): the default value is False. If it is True, the permutation histgram will be generated
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

    
    # visualization
    if visualization == True:
        print(f"The P-value of the Permutation Test is: {p_value}")
        plt.figure(figsize=(9,8))
        sns.histplot(data=box, bins='auto')
        plt.axvline(x=np.round(permu_mean,3), label='Permutation Mean at {}'.format(np.round(permu_mean,3)),c='g')
        plt.axvline(x=orig_mean, label='Original Mean at {}'.format(orig_mean), c='r', linestyle = 'dashed')
        plt.legend()
        plt.show()
    return p_value
    

#############################
### Test codes:
# x = np.random.random_sample((20,))
# y = np.random.random_sample((12,))
# x = [1,2,3,4,5]
# y = [6,7]
# xy = x+y
# print(PermutationTest(x, y, 10000, False))
#############################



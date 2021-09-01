# !/usr/bin/python

import numpy as np
import scipy.stats
import pandas as pd
"""
The contrast analysis used for group comparison

Author: Yile Wang
Date: 08/17/2021
"""

def contrast_analysis(datatable, contrast):
    """ 
    Arg: 
        Pandas DataFrame with all the features and groups info
    Return: 
        The contrast analysis results
    
    For this dataset, it should contain four groups, SNC, NC, MCI, AD;


    """

    # the number of cases for each group
    num_group = len(contrast)
    num_cases = datatable.groupby(['groups']).count().iloc[:,0].to_numpy()

    F_table = pd.DataFrame(columns=['features','F_value', 'P_value'])
    mean_array = np.zeros(num_group)
    var_array = np.zeros(num_group)

    for col in datatable.columns[3:]:

        # mean calculation
        mean_array = datatable.groupby(['groups']).mean().loc[:,col].to_numpy()
        meanNcontrast = np.dot(mean_array, contrast)
        contrast2 = np.square(contrast)

        # variance calculation
        var_array = datatable.groupby(['groups']).var().loc[:,col].to_numpy()
        denominator = sum(num_cases) - num_group
        # degree of freedom of the each case
        num_cases_df = num_cases -1

        # compute the sum of squares & mean sum of squares 
        SSE = np.dot(var_array, num_cases_df)
        MSE = SSE/denominator
        tmp_ms_contrast = sum(contrast2/num_cases)

        # compute the MS contrast
        MS_contrast = (meanNcontrast**2) / tmp_ms_contrast
        F_value = MS_contrast/MSE
        F_critical_05 = scipy.stats.f.ppf(q=1-0.05, dfn=1, dfd=denominator)
        F_critical_01 = scipy.stats.f.ppf(q=1-0.01, dfn=1, dfd=denominator)
        F_critical_001 = scipy.stats.f.ppf(q=1-0.001, dfn=1, dfd=denominator)


        # for posterior contrast, using scheffe test
        scheffe = F_critical_05 * (num_group-1)
        if F_value >= scheffe:
            p = 0.05
        else:
            p = 'NA'

        print(f"The {col} contrast has F_value {F_value}, and the F_critical Scheffe's Test is {scheffe}")
        F_table = F_table.append({'features':col,'F_value':F_value, 'P_value':p}, ignore_index=True)
    return F_table
    


G_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx', sheet_name='Gc_Go')
Mix_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/mix_final.xlsx')

# G_table = pd.read_excel('/home/wayne/stat_data/Gc_Go.xlsx', sheet_name='Gc_Go')
# Mix_table = pd.read_excel('/home/wayne/stat_data/mix_final.xlsx')


contrast = [-3, -1, 1, 3]
contrast2 = [-2, -1, 1, 2]
contrast3 = [-1,3,-3,1]
F_table = contrast_analysis(Mix_table, contrast2)
print(F_table)






    
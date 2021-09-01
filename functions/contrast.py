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
    num_cases = datatable.groupby(['groups']).count().iloc[:,0].to_numpy()
    SNC_count = num_cases[0]
    NC_count = num_cases[1]
    MCI_count = num_cases[2]
    AD_count = num_cases[3]


    F_table = pd.DataFrame(columns=['features','F_value', 'P_value'])

    for col in datatable.columns[3:]:
        SNC_mean = datatable.groupby(['groups']).median().loc[:,col].to_numpy()[0]
        NC_mean = datatable.groupby(['groups']).median().loc[:,col].to_numpy()[1]
        MCI_mean = datatable.groupby(['groups']).median().loc[:,col].to_numpy()[2]
        AD_mean = datatable.groupby(['groups']).median().loc[:,col].to_numpy()[3]
        meanNcontrast = np.dot([SNC_mean, NC_mean, MCI_mean, AD_mean], contrast)
        contrast2 = np.square(contrast)


        SNC_var = datatable.loc[datatable['groups'].eq('1.SNC'), col].var()
        NC_var = datatable.loc[datatable['groups'].eq('2.NC'), col].var()
        MCI_var = datatable.loc[datatable['groups'].eq('3.aMCI'), col].var()
        AD_var = datatable.loc[datatable['groups'].eq('4.AD'), col].var()

        denominator = sum([SNC_count, NC_count, MCI_count, AD_count]) - 4

        SSE = sum([SNC_var*(SNC_count-1), NC_var*(NC_count-1), MCI_var*(MCI_count-1), AD_var*(AD_count-1)])

        MSE = SSE/denominator

        MS_contrast = (meanNcontrast**2) / (contrast2[0]/SNC_count + contrast2[1]/NC_count + contrast2[2]/MCI_count + contrast2[3]/AD_count)
        F_value = MS_contrast/MSE
        F_critical_05 = scipy.stats.f.ppf(q=1-0.05, dfn=1, dfd=denominator)
        F_critical_01 = scipy.stats.f.ppf(q=1-0.01, dfn=1, dfd=denominator)
        F_critical_001 = scipy.stats.f.ppf(q=1-0.001, dfn=1, dfd=denominator)


        if F_value >= F_critical_05:
            p = 0.05
        else:
            p = 'NA'

        print(f"The {col} contrast has F_value {F_value}, and the F_critical is {F_critical_05}")
        F_table = F_table.append({'features':col,'F_value':F_value, 'P_value':p}, ignore_index=True)
    return F_table
    


G_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx', sheet_name='Gc_Go')
Mix_table = pd.read_excel('C:/Users/Wayne/tvb/stat_data/mix_final.xlsx')
contrast = [-2,-1,1,2]
contrast2 = [-1,1,1,-1]
contrast3 = [-1,3,-3,1]
F_table = contrast_analysis(Mix_table, contrast2)
print(F_table)






    
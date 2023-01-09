import numpy as np
import pandas as pd


def LI_cal(file, keys, names=None):
    """
    A function to calculate the LI between left and right columns

    Parameters:
        file names
        keys: provide the column numbers of the dominant side (like, left, then we use left to subtract right)
        names (list or tuple): the new names you want to assign to the column variables 
    Return:
        files with LI metrics
    """
    table = pd.read_excel(file)
    colnames = table.columns.values.tolist()
    for i in keys:
        tmp_LI = np.abs(table.iloc[:,i] - table.iloc[:,i-1]) / (table.iloc[:,i] + table.iloc[:,i-1])
        table=pd.concat([table, tmp_LI], axis=1)
    if names == None:
        names = [*range(len(keys))]
    else:
        colnames.extend(names)
        table.columns = colnames
    return table
    


# example:

# filename = "/home/yat-lok/workspace/data4project/gc3mins/ampres.xlsx"
# keys = [3,5]
# table = LI_cal(filename, keys, ('LI_amp_gamma', 'LI_amp_theta'))
# print(table)
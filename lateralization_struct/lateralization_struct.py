import numpy as np
import pandas as pd
import time
import os



if __name__ == '__main__':
    grp_pools = ['AD', 'SNC', 'MCI', 'NC']
    start = time.time()
    pdList = []
    # fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(12,8))
    # fig.suptitle("G frequency and Gamma")
    # ax =0
    # col = ['b', 'r']
    # xx = 0
    pcgSC = pd.DataFrame(columns=['grp', 'caseid', 'sc'])
    for grp in grp_pools:
        # obtain the data path
        pth = 'C:/Users/Wayne/tvb/LFP/'+grp
        case_pools = os.listdir(pth)
        # iterate the case id.
        # color = col[xx]
        for caseid in case_pools:
            pth = 'C:/Users/Wayne/tvb/AUS/' + grp + '/' + caseid + '/weights.txt' 
            sc = pd.read_csv(pth, delimiter="\t", header=None)
            tmp = pd.DataFrame([[grp, caseid, sc.iloc[4, 5]]], columns=['grp', 'caseid', 'sc'])
            pcgSC = pcgSC.append(tmp, ignore_index=True)
    pcgSC.to_csv(r'C:/Users/Wayne/tvb/sc.csv', index=False, header=True)
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    grp_pools = [ 'SNC', 'NC', 'MCI', 'AD']
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
            tmp = pd.DataFrame([[grp, caseid, sc.iloc[14, 15]]], columns=['grp', 'caseid', 'sc'])
            pcgSC = pcgSC.append(tmp, ignore_index=True)
    fig = plt.figure(figsize=(10,10))
    plt.title('Homotopic-Streamlines')
    fig = sns.violinplot(x="grp", y="sc", data=pcgSC, capsize=.2,palette=["#66CDAA","#4682B4","#AB63FA","#FFA15A"])
    fig = sns.stripplot(x="grp", y="sc", data=pcgSC,color='black')
    fig = sns.pointplot(data=pcgSC, x='grp', y="sc", join=False, ci=None, color='red')
    # fig.set_ylim(-0.5, 1)
    # fig.set_yticks(np.arange(-0.5, 1, 0.1))
    plt.show()
    #pcgSC.to_csv(r'C:/Users/Wayne/tvb/sc.csv', index=False, header=True)
    
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # mix = pd.read_excel('C:/Users/Wayne/tvb/mix.xlsx', sheet_name="Sheet1")
    # sns.set_theme(style="whitegrid")
    # for i in list(mix.columns[3:]):
    #     fig = plt.figure(figsize=(10,10))
    #     plt.title(f'{i}')
    #     fig = sns.violinplot(x="grp", y=i, data=mix, capsize=.2, palette=["#66CDAA","#4682B4","#AB63FA","#FFA15A"])
    #     fig = sns.stripplot(x="grp", y=i, data=mix,color='black')
    #     fig = sns.pointplot(data=mix, x='grp', y=i, join=False, ci=None, color='red')
    #     # fig.set_ylim(-0.5, 1)
    #     # fig.set_yticks(np.arange(-0.5, 1, 0.1))
    #     tmp_name = [i,'.png']
    #     name = ''.join(tmp_name)
    #     plt.savefig(name)
    
    # delay
    # mix = pd.read_excel('C:/Users/Wayne/tvb/mix.xlsx', sheet_name="Sheet3")
    # fig = plt.figure(figsize=(10,10))
    # plt.title('delay')
    # fig = sns.violinplot(x="grp", y="delay", data=mix, capsize=.2, palette=["#66CDAA","#4682B4","#AB63FA","#FFA15A"])
    # fig = sns.stripplot(x="grp", y="delay", data=mix, hue = 'high', palette=['blue','black'] )
    # fig = sns.pointplot(data=mix, x='grp', y="delay", join=False, ci=None, color='red')
    # plt.show()
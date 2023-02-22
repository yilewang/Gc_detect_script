import matplotlib.pyplot as plt
import seaborn as sns
from statannotations import Annotator
import numpy as np
import sys
sys.path.append('/home/yat-lok/workspaces/tvbtools')


colorcoding = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
order = ['SNC', 'NC', 'MCI', 'AD']
pairs = [('SNC', 'AD'),('NC', 'AD'), ('MCI', 'AD'), ('SNC', 'MCI'), ('NC', 'MCI'), ('SNC', 'NC')]

### violin plot visualization
def violin_dot(data, x, y, stats_table, test_short_name='Permutation FDR', xlabel = 'Group', ylabel='Count', pairs=pairs, order=order):
    pvalues = np.array([stats_table[pairs[i]][y] for i in range(len(pairs))])
    sig_indices = np.where(np.array(pvalues)<0.05)[0]
    fig, ax = plt.subplots()
    sns.set_style("dark")#sns.set_theme()
    # plt.box(False) #remove box
    
    plt.tick_params(left=False, bottom=False) #remove ticks
    sns.violinplot(data = data, x=x, y = y, inner=None, bw=.4, palette=colorcoding, linewidth=3, alpha = 1)
    sns.stripplot(data = data, x=x, y=y, edgecolor="black", linewidth=2, palette=colorcoding, alpha = 0.7, zorder=1)
    sns.pointplot(data=data, x = x, y=y, estimator=np.mean, color="red", ci = None, join=False, markers='s', alpha=1)
    if len(sig_indices) > 0:
        sig_pvalues = pvalues[sig_indices].tolist()
        sig_pairs = [pairs[i] for i in sig_indices]
        annotator = Annotator(ax, pairs=sig_pairs, data=data, x=x, y=y, order=order, plot='violinplot')
        (annotator.configure(test=None, test_short_name=test_short_name).set_pvalues(pvalues=sig_pvalues).annotate())
    plt.xlabel(y)
    plt.ylabel(ylabel)
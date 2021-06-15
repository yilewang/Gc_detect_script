import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


filepath = 'C:/Users/Wayne/tvb/exp_trends.csv'
df = pd.read_csv(filepath, index_col=0)

slp = df.loc[:,'K']
Gcm = df.loc[:,'Gmax-Gc']

fig, axs = plt.subplots(1,2, tight_layout=True, figsize=(15,5))
fig.suptitle('Correlation between expoential K and G range')
grp = ['SNC','NC', 'MCI','AD']
col = ["#66CDAA","#4682B4","#AB63FA","#FFA15A"]
mak = ['*','o','^','s']
corrtmp = [0.743,0.450,0.458,0.337]
# corrdata = pd.DataFrame({'grp':grp, 'corr':corrtmp})
count = 0
for group in grp:
    ktmp = df.loc[[group], 'K']
    gtmp = df.loc[[group], 'Gmax-Gc']
    al1 = pd.concat([ktmp, gtmp], axis=1)
    al2 = al1.sort_values('Gmax-Gc')
    axs[0].plot(al2.loc[:,'Gmax-Gc'], -1*al2.loc[:,'K'],marker= mak[count],linestyle = '',color = col[count], label = group)
    count +=1
axs[0].set_xlabel('G range (Gmax - Gcritical)')
axs[0].set_ylabel('K parameter in Exp function')
axs[0].legend()
axs[1].bar(grp, height=corrtmp, color = col, label = '')
axs[1].set_xlabel('Groups')
axs[1].set_ylabel('Correlation Coeficients')
for index, value in enumerate(corrtmp):
    axs[1].text(index-0.1, value, str(value)+'**')

plt.show()
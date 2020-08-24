
from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import numpy as np
import tvb.simulator.plot.timeseries_interactive as ts_int
import warnings
import pandas as pd
import os
from joblib import Parallel, delayed
import multiprocessing
import _pickle as cPickle
from matplotlib.pyplot import subplot, figure
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


#os.chdir('C:/fft/AD')



LOG = get_logger('demo')
warnings.filterwarnings('ignore')
Gc = 'C:/tvb3.0/Gc.xlsx'
Go = 'C:/tvb3.0/Go.xlsx'

case_go = pd.read_excel(Gc)
case_go.head()

SNC_ID = case_go.values[0:10, 0]
NC_ID = case_go.values[10:26, 0]
MCI_ID = case_go.values[26:61, 0]
AD_ID = case_go.values[61:74, 0]
SNC_GO = case_go.values[0:10, 1]
NC_GO = case_go.values[10:26, 1]
MCI_GO = case_go.values[26:61, 1]
AD_GO = case_go.values[61:74, 1]

speed = 10.
data = []
t = []
data_branch = []
path = []
#pth1 = 'C:/fft/'
pth1 = 'C:/fft/'
#pth2 = 'SuperNormal/'
#pth2 = 'NormalControls/'
pth2 = 'NC/data_signal_'
#pth2 = 'AD/'
pth3 ='_Gc_'
pth4 = '.csv'
peak = []
Gc = []
stepsize = 0.001#setting stepsize of G value
startpoint = 0.001#start point
endpoint = 0.031#end point, generally 0.061
gcs = np.arange(startpoint, endpoint, stepsize)
gcs = np.around(gcs, decimals=3)
num_cores = multiprocessing.cpu_count()


cat_ID = NC_ID



def Gc_detect(A): #detect local minimum
    global sa
    sa = np.argmin(np.diff(A))
    grad = np.diff(A)
    grad2 = np.diff(A)
    local_min_group = argrelextrema(grad, np.less)[0].tolist()
    local_min = np.argmin(grad)
    grad2[int(local_min)] = np.max(local_min_group)
    second_local_min = np.argmin(grad2)
    grad2[int(second_local_min)] = np.max(local_min_group)
    third_local_min = np.argmin(grad2)
    sa = sa +1
    second_local_min = second_local_min + 1
    third_local_min = third_local_min + 1
    if second_local_min < 29:
        if A[second_local_min -1] - A[second_local_min+1] > A[sa -1] - A[sa] and abs(second_local_min - sa) > 1:
            sa = second_local_min
        elif A[second_local_min -2] - A[second_local_min] > A[sa -1] - A[sa] and abs(second_local_min - sa) > 1:
            sa = second_local_min

    if A[sa - 1] - A[sa - 2] > 1:
        sa = second_local_min

    if sa < 5 and second_local_min > 5:
        sa = second_local_min
    elif sa < 5 and second_local_min < 5 and third_local_min > 5:
        sa = third_local_min

    if A[sa+1] - A[sa] < -0.05 and A[sa + 2] > A[sa+1]:
        sa = sa + 1
    elif A[sa+1] - A[sa] < -0.05 and A[sa + 2] < A[sa+1] and A[sa + 3] > A[sa+2]:
        sa = sa + 2

    plt.plot(gcs, A)
    plt.plot(gcs[sa], A[sa], "x")
    plt.xticks(gcs)
    plt.title(cat_ID[y])
    plt.show()

    sa = sa * 0.001
    sa = sa + 0.001
    return sa


def var_calculation(cat, num):
    df = pd.read_csv(pth1 + pth2 +str(cat)+ pth3 + str(num) + pth4)
    df.head()
    arr = df.to_numpy()
    arr1 = arr[:, 1:5]
    arr2 = arr[:, 6:17]
    arr = np.concatenate((arr1, arr2), axis=1)
    var_si = np.zeros(81920)
    for i in range(81920):
        var_si[i] = np.var(arr[i,:])
    var_max[z] = np.max(var_si)


var_max = np.zeros(len(gcs))
Gcr = np.zeros(len(cat_ID))
for y in [11]:
    figure(y)
    for z in range(len(gcs)):
        var_calculation(cat_ID[y], gcs[z])
    Gcr[y] = Gc_detect(var_max)

Gc = pd.Series(Gcr)
ID = pd.Series(cat_ID)

frame = {'ID': ID, 'Gc': Gc}
G = pd.DataFrame(frame)
G.round(3)
os.chdir('C:/fft')
print(G)
G.to_csv(str(pth2[0:2])+'.csv')



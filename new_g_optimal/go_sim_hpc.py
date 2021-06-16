from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import warnings
import numpy as np
import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import time
import logging
import sys

"""
A python script to use SJ3D model for brain simulation
"""

grp = str(sys.argv[1])
caseid = str(sys.argv[2])
g = float(sys.argv[3]) 

# file is the connectivity path, go is the G value (such as, G = 0.015; G = 0.027)
def tvb_simulation(file, go):
    my_rng = np.random.RandomState(seed=42)
    oscillator = ReducedSetHindmarshRose()
    white_matter = connectivity.Connectivity.from_file(file)
    oscillator.variables_of_interest = ["xi"]
    white_matter.speed = np.array([speed])
    white_matter_coupling = coupling.Linear(a=go)
    # if the sampling hz is 81920, dt = 0.01220703125
    # if the sampling hz is 208, dt = 4.8076923076923
    heunint = integrators.HeunStochastic(dt= 0.01220703125, noise=noise.Additive(nsig=np.array([0.00001]), ntau=0.0, # feel free to edit the specific paramter value. The default nsig = 1.0
                                                                                random_stream=my_rng))
    monitors_Bold = (monitors.Bold(hrf_kernel = equations.Gamma(), period = 2000.0), 
                    monitors.TemporalAverage(period=1.0),
                    monitors.ProgressLogger(period = 1e3))
    sim = simulator.Simulator(model=oscillator, connectivity=white_matter, coupling=white_matter_coupling,
                              integrator=heunint, monitors=monitors_Bold, simulation_length=(10**4))
    sim.configure()

    (bold_time, bold_data), (tavg_time, tavg_data), _ = sim.run()

    #plt.figure(figsize=(10,5))
    #plt.plot(bold_time * ( 10**(-3) ), bold_data[:,0,:,0], alpha = 0.3)
    #plt.title("BOLD signals of limbic areas")
    #plt.xlabel("Time (s)")
    #plt.grid(True)
    #plt.show()
    return bold_data




speed = 10.




# #bash obtain the input
# parser = argparse.ArgumentParser(description='pass a float')
# parser.add_argument('float',type=float, help='the number')
# args = parser.parse_args()
# y = args.float

if __name__ == "__main__":
    test_file = '/mnt/c/Users/Wayne/workdir/zip/'+grp+'_conn/'+caseid+'.zip'
    #test_file = '/home/yxw190015/go/'+grp+'_conn/'+casid+'.zip'
    #test_file = 'C:/Users/Wayne/workdir/zip/'+grp+'_conn/'+caseid+'.zip'
    start_time = time.time()
    raw = tvb_simulation(test_file, np.array([g]))
    end_time = time.time()
    logging.warning('Duration: {}'.format(end_time - start_time))


    # plotting
    df = pd.DataFrame(raw[:, 0, :, 0], columns = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R'])
    save_path = '/mnt/c/Users/Wayne/workdir/output/'+grp+'/'+caseid+'/'+caseid+'_'+str(g)+'.csv'
    df.to_csv(save_path)
    #csv_file = 'go_'+str(y)+'.csv'
    #corrMatrix.to_csv(csv_file)
    #sn.heatmap(corrMatrix, annot = False, cmap= 'viridis')
    #plt.show()
    ## limbic = np.array(raw)
    # PCG_L = limbic[:,0, 4, 0]
    # PCG_R = limbic[:, 0, 5, 0]
    # PCG = {'PCG_R': PCG_R, 'PCG_L': PCG_L}
    # df = pd.DataFrame(PCG)
    # csv_file = '/home/yxw190015/go/Go_'+str(y)+'.csv'
    # df.to_csv(csv_file) 





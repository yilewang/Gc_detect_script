
from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import warnings
import numpy as np
import argparse
import pandas as pd
import time
import logging
import sys
import multiprocessing


# grp = str(sys.argv[1])
# caseid = str(sys.argv[2])
# g = float(sys.argv[3]) 

def tvb_simulation(file, gm):
    my_rng = np.random.RandomState(seed=42)
    conn=connectivity.Connectivity.from_file(file)
    conn.speed=np.array([10.0])
    sim = simulator.Simulator(
        model=ReducedSetHindmarshRose(), 
        connectivity=conn,
        coupling=coupling.Linear(a=np.array([gm])),
        simulation_length=1e3,
        integrator=integrators.HeunStochastic(dt=0.01220703125, noise=noise.Additive(nsig=np.array([0.00001]), ntau=0.0,
                                                                                random_stream=np.random.RandomState(seed=42))),
        monitors=(
            monitors.Raw(),
            monitors.TemporalAverage(period=1.),
            monitors.ProgressLogger(period=1e2)
        )
    ).configure()

    (raw_time, raw_data), (tavg_time, tavg_data), _ = sim.run()
    df = pd.DataFrame(raw_data[:, 0, :, 0], columns = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R'])
    save_path = '/mnt/c/Users/Wayne/tvb/TVB_workflow/new_g_max/'+grp+'/'+file[-9:-4]+'/'+file[-9:-4]+'_'+str(gm)+'.csv'
    df.to_csv(save_path)

    #plt.figure(figsize=(10,5))
    #plt.plot(bold_time * ( 10**(-3) ), bold_data[:,0,:,0], alpha = 0.3)
    #plt.title("BOLD signals of limbic areas")
    #plt.xlabel("Time (s)")
    #plt.grid(True)
    #plt.show()





# #bash obtain the input
# parser = argparse.ArgumentParser(description='pass a float')
# parser.add_argument('float',type=float, help='the number')
# args = parser.parse_args()
# y = args.float

if __name__ == "__main__":
    grp_pools = ['AD', 'MCI','NC','SNC']
    for grp in grp_pools:
        pth = '/mnt/c/Users/Wayne/output/'+grp
        case_pools = os.listdir(pth)
        dataPth = []
        for caseid in case_pools:
            dataFile = '/mnt/c/Users/Wayne/workdir/zip/'+grp+'_conn/'+caseid+'.zip'
            dataPth.append(dataFile)
            #test_file = '/home/yxw190015/go/'+grp+'_conn/'+casid+'.zip'
            #test_file = 'C:/Users/Wayne/workdir/zip/'+grp+'_conn/'+caseid+'.zip'



        glist = np.arange(0.001, 0.08, 0.001)
        g = np.around(glist, 3)
        start_time = time.time()
        # get cores
        cores = multiprocessing.cpu_count()
        # start processes
        pool = multiprocessing.Pool(processes=cores)
        tasks = [(file, gm) for file in dataPth for gm in g]
        pool.starmap(tvb_simulation, tasks)
        #raw = tvb_simulation(dataFile, np.array([g]))
        end_time = time.time()
        logging.warning('Duration: {}'.format(end_time - start_time))



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

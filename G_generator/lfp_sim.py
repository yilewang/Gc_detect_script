
from py import process
from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import warnings
import numpy as np
import pandas as pd
import time
import logging
import sys
sys.path.insert(0, 'C:/Users/Wayne/tvb/TVB_workflow/functions/')
import multiprocessing as mp
# pool = mp.Pool(mp.cpu_count())
import tvb_sim
import functools
import joblib


if __name__ ==  '__main__':
    coData = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx', index_col=0)
    grp_pools = list(coData.groups)
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    case_pools = list(coData.index)

    task = [(caseid, grp, np.round(coData.loc[caseid, "Gc"], 3)) for grp in grp_pools for caseid in case_pools]
    pool.starmap(tvb_sim.tvb_simulation, task)

    # for grp in grp_pools:
    #     try:
            
    #         # iterate the case id.
    #         funcs = [functools.partial(tvb_sim.tvb_simulation, caseid, grp, np.array(np.round(coData.loc[caseid, "Gc"], 3))) for caseid in case_pools]

    #         # three parameters in tvb_simulation(caseid, group info, scaling factor global coupling)
    #         joblib.Parallel(n_jobs=4)(joblib.delayed(tvb_sim.tvb_simulation(caseid, grp, np.array(np.round(coData.loc[caseid, "Gc"], 3)))) for caseid in case_pools)

    #         # with mp.Pool() as p:
    #         #     [p.apply(tvb_sim.tvb_simulation, args=(caseid, grp, np.array(np.round(coData.loc[caseid, "Gc"], 3)))) for caseid in case_pools]
    #         #     p.close()
    #     except FileNotFoundError:
    #         continue
    #     except KeyError:
    #         continue


        # case_pools = list(coData.index)
        # for caseid in case_pools:
        #     try:
        #         gm = np.round(coData.loc[caseid, "Gc"], 3)
        #         #test_file = 'C:/Users/Wayne/workdir/zip/'+grp+'_conn/'+caseid+'.zip'
        #         start_time = time.time()
        #         # raw = pool.starmap(tvb_sim.tvb_simulation, [(caseid, grp, globalCoupling) for caseid in case_pools])
        #         # pool.close()
        #         tvb_sim.tvb_simulation(caseid, grp, gm)
        #         end_time = time.time()
        #         logging.warning('Duration: {}'.format(end_time - start_time))
        #         # df = pd.DataFrame(raw[:, 0, :, 0], columns = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R'])

        #         # save_path='C:/Users/Wayne/tvb/output/'+caseid+'_'+str(gm)+'.csv'
        #         # df.to_csv(save_path)
        #     except FileNotFoundError:
        #         continue
        #     except KeyError:
        #         continue


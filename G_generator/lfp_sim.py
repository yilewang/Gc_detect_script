if __name__ ==  '__main__':
    from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
    from tvb.simulator.lab import *
    import warnings
    import numpy as np
    import pandas as pd
    # import time
    # import logging
    import sys
    sys.path.insert(0, 'C:/Users/Wayne/tvb/TVB_workflow/functions/')
    import multiprocess as mp
    # pool = mp.Pool(mp.cpu_count())
    import tvb_sim
    import functools
    import joblib

    coData = pd.read_excel('C:/Users/Wayne/tvb/stat_data/Gc_Go.xlsx', index_col=0)
    grp_pools = ['SNC','NC','MCI', 'AD']
    for grp in grp_pools:
        try:
            case_pools = list(coData.index)
            # iterate the case id.
            # funcs = [functools.partial(tvb_sim.tvb_simulation, caseid, grp, np.array(np.round(coData.loc[caseid, "Gc"], 3))) for caseid in case_pools]

            # three parameters in tvb_simulation(caseid, group info, scaling factor global coupling)
            joblib.Parallel(n_jobs=4)(joblib.delayed(tvb_sim.tvb_simulation(caseid, grp, np.array(np.round(coData.loc[caseid, "Gc"], 3)))) for caseid in case_pools)

            # with mp.Pool() as p:
            #     [p.apply(tvb_sim.tvb_simulation, args=(caseid, grp, np.array(np.round(coData.loc[caseid, "Gc"], 3)))) for caseid in case_pools]
            #     p.close()
        except FileNotFoundError:
            continue
        except KeyError:
            continue







            # for caseid in case_pools:
            #     try:
            #         gm = np.round(coData.loc[caseid, "Gc"], 3)
            #         #test_file = 'C:/Users/Wayne/workdir/zip/'+grp+'_conn/'+caseid+'.zip'
            #         # start_time = time.time()
            #         globalCoupling = np.array([gm])
            #         raw = pool.starmap(tvb_sim.tvb_simulation, [(caseid, grp, globalCoupling) for caseid in case_pools])
            #         pool.close()
            #         # raw = tvb_simulation(test_file, globalCoupling)
            #         # end_time = time.time()
            #         # logging.warning('Duration: {}'.format(end_time - start_time))
            #         df = pd.DataFrame(raw[:, 0, :, 0], columns = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R'])

            #         save_path='C:/Users/Wayne/tvb/output/'+caseid+'_'+str(gm)+'.csv'
            #         df.to_csv(save_path)
            #     except FileNotFoundError:
            #         continue
            #     except KeyError:
            #         continue


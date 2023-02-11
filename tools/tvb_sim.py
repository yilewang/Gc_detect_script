from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def tvb_simulation(gc):

    # file path of tvb connectome zip file
    # conduction velocity
    connectivity.speed = np.array([10.])
    # simulation parameters
    sim = simulator.Simulator(
    model=ReducedSetHindmarshRose(),
    connectivity=connectivity.Connectivity.from_file(),
    coupling=coupling.Linear(a=np.array([gc])),
    simulation_length=1e4,
    integrator=integrators.HeunStochastic(dt=0.01220703125, noise=noise.Additive(nsig=np.array([0.00001]), ntau=0.0,
                                                                                random_stream=np.random.RandomState(seed=42))),
    monitors=(
    monitors.TemporalAverage(period=1.),
    monitors.Raw(),
    monitors.ProgressLogger(period=1e2))
    ).configure()

    sim.configure()

    # run the simulation
    (tavg_time, tavg_data), (raw_time, raw_data),_ = sim.run()

    # write it into the csv file
    # df = pd.DataFrame(raw_data[:, 0, :, 0], columns = ['aCNG-L', 'aCNG-R','mCNG-L','mCNG-R','pCNG-L','pCNG-R', 'HIP-L','HIP-R','PHG-L','PHG-R','AMY-L','AMY-R', 'sTEMp-L','sTEMP-R','mTEMp-L','mTEMp-R'])
    df = pd.DataFrame(raw_data[:, 0, :, 0])
    
    return df


# file = "/Users/yat-lok/Desktop/0306A.zip"
sim = tvb_simulation(0.015)
plt.plot(sim)
plt.show()
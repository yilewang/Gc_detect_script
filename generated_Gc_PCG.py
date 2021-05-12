
from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import numpy as np
import tvb.simulator.plot.timeseries_interactive as ts_int
import warnings
import pandas as pd
import os
import _pickle as cPickle

os.chdir('W:/inter_hemi/AD')



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
pth1 = 'C:/Users/Wayne/Desktop/AUS-76-DATASET/ALL_Structure_category/Joelle_normalized/'
pth2 = 'AD/'
#pth2 = 'NormalControls/'
#pth2 = 'MCI/'
#pth2 = 'AD/'
pth3 = '.zip'
peak = []

cat_ID = AD_ID
cat_GO = AD_GO

def csgo(file, go):
    my_rng = np.random.RandomState(seed=42)
    oscillator = ReducedSetHindmarshRose()
    white_matter = connectivity.Connectivity.from_file(file)
    oscillator.variables_of_interest = ["xi"]
    white_matter.speed = np.array([speed])
    white_matter_coupling = coupling.Linear(a=np.array([go]))
    heunint = integrators.HeunStochastic(dt=0.01220703125, noise=noise.Additive(nsig=np.array([0.00001]), ntau=0.0,
                                                                                random_stream=my_rng))
    mon_raw = monitors.Raw()
    mon_tavg = monitors.TemporalAverage(period=1.)
    what_to_watch = (mon_raw, mon_tavg)
    sim = simulator.Simulator(model=oscillator, connectivity=white_matter, coupling=white_matter_coupling,
                              integrator=heunint, monitors=what_to_watch)
    sim.configure()

    raw_data = []
    raw_time = []
    tavg_data = []
    tavg_time = []

    for raw, tavg in sim(simulation_length=1000):
        if not raw is None:
            raw_time.append(raw[0])
            raw_data.append(raw[1])

        if not tavg is None:
            tavg_time.append(tavg[0])
            tavg_data.append(tavg[1])

        # Make the lists numpy.arrays for easier use.
        RAW = np.array(raw_data)
        data.append(RAW[:, 0, :, 0])
        t = raw_time

    # sim_state_fname = 'sim_state.pickle'
    #
    # with open(sim_state_fname, 'wb') as file_descr:
    #     cPickle.dump({
    #         'history': sim.history.buffer,
    #         'current_step': sim.current_step,
    #         'current_state': sim.current_state,
    #         'rng': sim.integrator.noise.random_stream.get_state()}, file_descr)
    # file_descr.close()
    # del sim
    #
    # sim = simulator.Simulator(model=oscillator, connectivity=white_matter,
    #                           coupling=white_matter_coupling,
    #                           integrator=heunint, monitors=what_to_watch)
    # sim.configure()
    #
    # with open(sim_state_fname, 'rb') as file_descr:
    #     while True:
    #         try:
    #             state = cPickle.load(file_descr)
    #         except:
    #             break
    #     sim.history.buffer = state['history']
    #     sim.current_step = state['current_step']
    #     sim.current_state = state['current_state']
    #     sim.integrator.noise.random_stream.set_state(state['rng'])
    #
    # raw_data_branch = []
    # raw_time_branch = []
    # tavg_data_branch = []
    # tavg_time_branch = []
    #
    # for raw, tavg in sim():
    #     if not raw is None:
    #         raw_time_branch.append(raw[0])
    #         raw_data_branch.append(raw[1])
    #
    #     if not tavg is None:
    #         tavg_time_branch.append(tavg[0])
    #         tavg_data_branch.append(tavg[1])
    #
    # RAW_branch = np.array(raw_data_branch)
    # data_branch.append(RAW_branch[:, 0, :, 0])
    # t = raw_time_branch
    #
    # # create and launch the interactive visualiser
    # tsr = time_series.TimeSeriesRegion(data=np.array(raw_data_branch), connectivity=sim.connectivity,
    #                                    sample_period=sim.monitors[0].period / 1e3,
    #                                    sample_period_unit='s')
    # tsr.configure()
    # tsr
    # tsi = ts_int.TimeSeriesInteractive(time_series=tsr)
    # tsi.configure()
    #
    # RAW = np.array(tsr.data)
    # TR1 = RAW[:, 0, 5, 0]  # PCG_RIGHT
    # TR2 = RAW[:, 0, 4, 0]  # PCG_LEFT
    # TR = {'PCG_R': TR1, 'PCG_L': TR2}
    # df = pd.DataFrame(TR)
    # df.to_csv('data_signal_' + str(cat_ID[y]) + '_Gc.csv')

for y in range(len(cat_ID)):
    pth = pth1 + pth2 + cat_ID[y] + pth3
    path.append(pth)
    csgo(path[y], cat_GO[y])


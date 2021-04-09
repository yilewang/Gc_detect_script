from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
from tvb.simulator.lab import *
import warnings
import numpy as np


def tvb_simulation(file, go):
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

    for raw, tavg in sim():
        if not raw is None:
            raw_time.append(raw[0])
            raw_data.append(raw[1])

        if not tavg is None:
            tavg_time.append(tavg[0])
            tavg_data.append(tavg[1])

        t = raw_time

    return raw_data



go_range = [0.001]
speed = 10.
if __name__ == "__main__":
    for y in range(len(go_range)):
        #test_file = '/home/yxw190015/go/0306A.zip'
        test_file = 'C:/Users/Wayne/OneDrive - The University of Texas at Dallas/AUS-76-DATASET/ALL_Structure_category/Joelle_normalized/AD/dd/0306A.zip'
        raw = tvb_simulation(test_file, y)
        print(np.shape(raw))



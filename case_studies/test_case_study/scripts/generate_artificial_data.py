import numpy as np
import xarray as xr

from pymob.utils.store_file import prepare_casestudy

config = prepare_casestudy(("test_case_study", "test_scenario"), "settings.cfg")
from test_case_study.sim import Simulation

rng = np.random.default_rng(871237)


sim = Simulation(config=config)
sim.compute()
Y = sim.Y
X = sim.coordinates["time"]

data_index = rng.choice(a=np.arange(len(X)), size=100, replace=False)
data_index.sort()
Y_stochastic = rng.poisson(Y)[data_index, :]
X_stochastic = X[data_index]

# save noisy dataset
noisy_dataset = sim.create_dataset_from_numpy(
    Y_stochastic, ["rabbits", "wolves"], {"time": X_stochastic}
)
noisy_dataset.to_netcdf(f"{sim.data_path}/simulated_noisy_data.nc")

sim.results.to_netcdf(f"{sim.data_path}/simulated_data.nc")


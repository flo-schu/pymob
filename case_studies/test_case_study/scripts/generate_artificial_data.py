import numpy as np
import xarray as xr

from pymob.utils.store_file import prepare_casestudy

config = prepare_casestudy(("test_case_study", "test_scenario"), "settings.cfg")
from test_case_study.sim import Simulation

rng = np.random.default_rng(871237)


sim = Simulation(config=config)
evaluator = sim.dispatch(theta=sim.model_parameter_dict)
evaluator()

# save dataset
evaluator.results.to_netcdf(f"{sim.data_path}/simulated_data.nc")

# add noise to daa
Y = evaluator.results.copy()
X = sim.coordinates["time"].copy()

data_index = rng.choice(a=np.arange(len(X)), size=200, replace=False)
data_index.sort()
X_stochastic = X[data_index]
y_noisy = Y.isel(time=data_index)

nan_frac = 0.1
rng = np.random.default_rng(seed=1)
for i, (k, val) in enumerate(y_noisy.items()):
    val_stoch = rng.lognormal(np.log(val) + 1e-8, sigma=0.1)
    nans = rng.binomial(n=1, p=nan_frac, size=val_stoch.shape)
    val.values = np.where(nans == 1, np.nan, val_stoch)


# save noisy dataset
y_noisy.to_netcdf(f"{sim.data_path}/simulated_noisy_data.nc")



import os
import click

from lotka_volterra_UDE_case_study.mod import Func
import jax.random as jrandom
import jax.numpy as jnp
import xarray as xr
from pymob.simulation import SimulationBase
from pymob.solvers.diffrax import UDESolver
from pymob.sim.config import Param
import diffrax
import jax.random as jr
import jax
import arviz as az
import numpy as np
import pysindy as ps
import re

jax.config.update('jax_enable_x64', True)

@click.command()
@click.option("-length", "--length_strategy", type=(float, float, float, float), default=(0.1, 1, -1, -1))
@click.option("-lr", "--lr_strategy", type=float, default=1e-3)
@click.option("-clip", "--clip_strategy", type=float, default=0.1)
@click.option("-batch", "--batch_size", type=int, default=20)
@click.option("-points", "--data_points", type=int, default=51)
@click.option("-noise", "--data_noise", type=float, default=0.0)
def main(length_strategy, lr_strategy, clip_strategy, batch_size, data_points, data_noise):

    string = ""

    try: 

        idata = az.from_netcdf(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}/optax_idata.nc")

        n = 50

        # Initialize the simulation object
        sim = SimulationBase()

        # Configure the case study
        sim.config.case_study.name = "lotka_volterra_UDE_case_study"
        sim.config.case_study.scenario = "UDETest"

        # Add the model to the simulation
        key = jrandom.PRNGKey(5678)
        data_key, model_key, loader_key = jrandom.split(key, 3)
        sim.model = Func({"alpha":jnp.array(1.3), "delta":idata.posterior.sel(draw=0).delta.values[0]},weights=idata.posterior.sel(draw=0).weights.values.tolist()[0],bias=idata.posterior.sel(draw=0).bias.values.tolist()[0],key=model_key)

        # Define a solver
        sim.solver = UDESolver

        # Add our dataset to the simulation
        sim.observations = idata.observed_data.sel(batch_id = slice(39.9,50))

        # Add the initial condition to the simulation
        sim.model_parameters["y0"] = sim.observations.sel(time = 0).drop_vars("time")

        sim.config.model_parameters.alpha = Param(value=1.3, free=False)
        sim.config.model_parameters.delta = Param(value=1.8, free=True)
        sim.config.model_parameters.delta.prior = "uniform(loc=1.0,scale=2.0)"

        sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

        # Create an xArray dataset containing the external input data
        # xin = xr.DataArray(np.zeros(201), coords={"time": ts}).to_dataset(name="x_in")

        # Add external inputs to the simulation
        # sim.model_parameters["x_in"] = xin

        sim.config.jaxsolver.max_steps = 10000
        sim.config.jaxsolver.throw_exception = False

        sim.coordinates["time"] = np.linspace(0,50,1001)

        # Put everything in place for running the simulation
        sim.dispatch_constructor()

        # Create an evaluator, run the simulation and obtain the results
        evaluator = sim.dispatch()

        successes = 0
        equations = []

        ts = np.linspace(0,50,1001)

        for (i,m) in enumerate(idata.posterior.draw.values):

            evaluator.model = Func({"alpha":jnp.array(1.3), "delta":idata.posterior.sel(draw=m).delta.values[0]},weights=idata.posterior.sel(draw=m).weights.values.tolist()[0],bias=idata.posterior.sel(draw=m).bias.values.tolist()[0],key=model_key)
            evaluator()

            res = [jnp.stack(jnp.array([evaluator.Y["prey"][j], evaluator.Y["predator"][j]]),axis=1) for j in range(10)]

            dervs = [jnp.array([evaluator.model(None, y, (), None) for y in ys]) for ys in res]

            for j in range(10):
                
                print(str(i) + ": " + str(j))

                try:
                    psmodel = ps.SINDy()
                    psmodel.fit(res[j], t=ts, x_dot=dervs[j], feature_names=["X","Y"])
                    eqs = psmodel.equations()
                    equations.append(eqs)
                    success = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                except ValueError:
                    success = False
                    print("error")

                successes += success

        string = str(idata.posterior.draw.values.shape[0])+";"+str(successes)+"\n"

        for eq in equations:
            string = string + str(eq[0]) + "\n" + str(eq[1]) + "\n"

    except FileNotFoundError:

        string = "0;0"

    with open(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}/SINDy_results.txt","w") as variable_name:
        variable_name.write(string)

if __name__ == "__main__":
    main()
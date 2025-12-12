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

        idata = az.from_netcdf(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/optax_idata.nc")

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
        successes2 = 0
        successes3 = 0
        successesp = 0
        successesp2 = 0
        successesp3 = 0
        equations = []

        successes_2 = 0
        successes2_2 = 0
        successes3_2 = 0
        successesp_2 = 0
        successesp2_2 = 0
        successesp3_2 = 0
        equations_2 = []

        ts = np.linspace(0,50,1001)

        for (i,m) in enumerate(idata.posterior.draw.values):

            evaluator.model = Func({"alpha":jnp.array(1.3), "delta":idata.posterior.sel(draw=m).delta.values[0]},weights=idata.posterior.sel(draw=m).weights.values.tolist()[0],bias=idata.posterior.sel(draw=m).bias.values.tolist()[0],key=model_key)
            evaluator()

            res = [jnp.stack(jnp.array([evaluator.Y["prey"][j], evaluator.Y["predator"][j]]),axis=1) for j in range(10)]

            dervs = [jnp.array([evaluator.model(None, y, (), None) for y in ys]) for ys in res]

            for j in range(10):
                
                print(str(i) + ": " + str(j))

                try:
                    psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.3))
                    psmodel.fit(res[j], t=ts, x_dot=dervs[j], feature_names=["X","Y"])
                    eqs = psmodel.equations()
                    eqs.append(False)
                    equations.append(eqs)
                    success = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                    success_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                    success_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                    success2 = success_t1 and success_t2
                    success_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None)
                    success_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None)
                    success3 = success_t1 and success_t2
                    psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.5))
                    psmodel.fit(res[j], t=ts, x_dot=dervs[j], feature_names=["X","Y"])
                    eqs = psmodel.equations()
                    eqs.append(False)
                    equations_2.append(eqs)
                    success_2 = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                    success_t1_2 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                    success_t2_2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                    success2_2 = success_t1_2 and success_t2_2
                    success_t1_2 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None)
                    success_t2_2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None)
                    success3_2 = success_t1_2 and success_t2_2
                except ValueError:
                    success = False
                    success2 = False
                    success3 = False
                    success_2 = False
                    success2_2 = False
                    success3_2 = False
                    print("error")

                successes += success
                successes2 += success2
                successes3 += success3
                successes_2 += success_2
                successes2_2 += success2_2
                successes3_2 += success3_2

            try:
                psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.3))
                psmodel.fit(np.array(res), t=ts, x_dot=np.array(dervs), feature_names=["X","Y"])
                eqs = psmodel.equations()
                eqs.append(True)
                equations.append(eqs)
                successp = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                successp_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                successp_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                successp2 = successp_t1 and successp_t2
                successp_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None)
                successp_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None)
                successp3 = successp_t1 and successp_t2
                psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.5))
                psmodel.fit(np.array(res), t=ts, x_dot=np.array(dervs), feature_names=["X","Y"])
                eqs = psmodel.equations()
                eqs.append(True)
                equations_2.append(eqs)
                successp_2 = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                successp_t1_2 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                successp_t2_2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                successp2_2 = successp_t1_2 and successp_t2_2
                successp_t1_2 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None)
                successp_t2_2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None)
                successp3_2 = successp_t1_2 and successp_t2_2
            except ValueError:
                successp = False
                successp2 = False
                successp3 = False
                successp_2 = False
                successp2_2 = False
                successp3_2 = False
                print("error")

            successesp += successp
            successesp2 += successp2
            successesp3 += successp3
            successesp_2 += successp_2
            successesp2_2 += successp2_2
            successesp3_2 += successp3_2

        string = str(idata.posterior.draw.values.shape[0])+";"+str(successes)+";"+str(successes2)+";"+str(successes3)+";"+str(successesp)+";"+str(successesp2)+";"+str(successesp3)+"\n"
        string_2 = str(idata.posterior.draw.values.shape[0])+";"+str(successes_2)+";"+str(successes2_2)+";"+str(successes3_2)+";"+str(successesp_2)+";"+str(successesp2_2)+";"+str(successesp3_2)+"\n"

        for eq in equations:
            if eqs[2]:
                string = string + "pooled: " + str(eq[0]) + "\n" + str(eq[1]) + "\n"
            else:
                string = string + str(eq[0]) + "\n" + str(eq[1]) + "\n"
        for eq in equations_2:
            if eqs[2]:
                string_2 = string_2 + "pooled: " + str(eq[0]) + "\n" + str(eq[1]) + "\n"
            else:
                string_2 = string_2 + str(eq[0]) + "\n" + str(eq[1]) + "\n"

    except FileNotFoundError:

        string = "0;0;0;0;0;0;0"
        string_2 = "0;0;0;0;0;0;0"

    with open(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/SINDy_results.txt","w") as variable_name:
        variable_name.write(string)
    with open(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/SINDy_results_0.5.txt","w") as variable_name:
        variable_name.write(string_2)

if __name__ == "__main__":
    main()
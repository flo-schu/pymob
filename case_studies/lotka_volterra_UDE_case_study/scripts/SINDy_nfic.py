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

        # Load inference data
        idata = az.from_netcdf(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/optax_idata.nc")

        # Create simulation base and model
        sim = SimulationBase()
        sim.config.case_study.name = "lotka_volterra_UDE_case_study"
        sim.config.case_study.scenario = "UDETest"
        key = jrandom.PRNGKey(5678)
        data_key, model_key, loader_key = jrandom.split(key, 3)
        sim.model = Func({"alpha":jnp.array(1.3), "delta":idata.posterior.sel(draw=0).delta.values[0]},weights=idata.posterior.sel(draw=0).weights.values.tolist()[0],bias=idata.posterior.sel(draw=0).bias.values.tolist()[0],key=model_key)

        # Add dataset from inference data and initial conditions to the simulation
        sim.observations = idata.observed_data.sel(batch_id = slice(39.9,50))
        sim.model_parameters["y0"] = sim.observations.sel(time = 0).drop_vars("time")

        # Define model parameters
        sim.config.model_parameters.alpha = Param(value=1.3, free=False)
        sim.config.model_parameters.delta = Param(value=1.8, free=True)
        sim.config.model_parameters.delta.prior = "uniform(loc=1.0,scale=2.0)"
        sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

        # Set solver settings
        sim.solver = UDESolver
        sim.config.jaxsolver.max_steps = 10000
        sim.config.jaxsolver.throw_exception = False

        # Change time coordinate
        sim.coordinates["time"] = np.linspace(0,50,1001)

        # Put everything in place for running the simulation
        sim.dispatch_constructor()

        # Dispatch evaluator
        evaluator = sim.dispatch()

        # Prepare evaluation of SINDy results
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

        # Create dense time axis
        ts = np.linspace(0,50,1001)

        # For each model, run SINDy with two different thresholds and two
        # different approaches (individual vs. pooled time series) and
        # evaluate with three different rules.
        for (i,m) in enumerate(idata.posterior.draw.values):

            # Provide evaluator with model and run it
            evaluator.model = Func({"alpha":jnp.array(1.3), "delta":idata.posterior.sel(draw=m).delta.values[0]},weights=idata.posterior.sel(draw=m).weights.values.tolist()[0],bias=idata.posterior.sel(draw=m).bias.values.tolist()[0],key=model_key)
            evaluator()

            # Change structure of simulated time series and calculate derivatives
            res = [jnp.stack(jnp.array([evaluator.Y["prey"][j], evaluator.Y["predator"][j]]),axis=1) for j in range(10)]
            dervs = [jnp.array([evaluator.model(None, y, (), None) for y in ys]) for ys in res]

            # Run SINDy for each individual time series
            for j in range(10):
                
                print(str(i) + ": " + str(j))

                try:
                    # Create SINDy model for 0.3 threshold and run it
                    psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.3))
                    psmodel.fit(res[j], t=ts, x_dot=dervs[j], feature_names=["X","Y"])
                    
                    # Save equations provided by SINDy
                    eqs = psmodel.equations()
                    eqs.append(False)
                    equations.append(eqs)

                    # Evaluate rule 1
                    success = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                    # Evaluate rule 2
                    success_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                    success_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                    success2 = success_t1 and success_t2
                    # Evaluate rule 3
                    success_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None)
                    success_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None)
                    success3 = success_t1 and success_t2
                    
                    # Create SINDy model for 0.5 threshold and run it
                    psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.5))
                    psmodel.fit(res[j], t=ts, x_dot=dervs[j], feature_names=["X","Y"])
                    
                    # Save equations provided by SINDy
                    eqs = psmodel.equations()
                    eqs.append(False)
                    equations_2.append(eqs)

                    # Evaluate rule 1
                    success_2 = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                    # Evaluate rule 2
                    success_t1_2 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                    success_t2_2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                    success2_2 = success_t1_2 and success_t2_2
                    # Evaluate rule 3
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

                # Add success for this model and time series to overall number of successes
                successes += success
                successes2 += success2
                successes3 += success3
                successes_2 += success_2
                successes2_2 += success2_2
                successes3_2 += success3_2

            # Run SINDy for pooled time series
            try:
                # Create SINDy model for 0.3 threshold and run it
                psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.3))
                psmodel.fit(np.array(res), t=ts, x_dot=np.array(dervs), feature_names=["X","Y"])
                
                # Save equations provided by SINDy
                eqs = psmodel.equations()
                eqs.append(True)
                equations.append(eqs)

                # Evaluate rule 1
                successp = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                # Evaluate rule 2
                successp_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                successp_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                successp2 = successp_t1 and successp_t2
                # Evaluate rule 3
                successp_t1 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None)
                successp_t2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None)
                successp3 = successp_t1 and successp_t2
                
                # Create SINDy model for 0.5 threshold and run it
                psmodel = ps.SINDy(optimizer=ps.optimizers.STLSQ(threshold=0.5))
                psmodel.fit(np.array(res), t=ts, x_dot=np.array(dervs), feature_names=["X","Y"])
                
                # Save equations provided by SINDy
                eqs = psmodel.equations()
                eqs.append(True)
                equations_2.append(eqs)

                # Evaluate rule 1
                successp_2 = ((re.fullmatch('\d+\.\d+\sX\s\+\s\-\d+\.\d+\sX\sY',eqs[0]) != None or re.fullmatch('\-\d+\.\d+\sX\sY\s\+\s\d+\.\d+\sX',eqs[0]) != None) and (re.fullmatch('\-\d+\.\d+\sY\s\+\s\d+\.\d+\sX\sY',eqs[1]) != None or re.fullmatch('\d+\.\d+\sX\sY\s\+\s\-\d+\.\d+\sY',eqs[1]) != None))
                # Evaluate rule 2
                successp_t1_2 = ((re.search("\+\s\d+\.\d+\sX", eqs[0]) != None) or (re.match("\d+\.\d+\sX", eqs[0]) != None)) and (re.search("\-\d+\.\d+\sX\sY", eqs[0]) != None) and ((eqs[0].count("+")) <= 2)
                successp_t2_2 = ((re.search("\+\s\d+\.\d+\sX\sY", eqs[1]) != None) or (re.match("\d+\.\d+\sX\sY", eqs[1]) != None)) and (re.search("\-\d+\.\d+\sY", eqs[1]) != None) and ((eqs[1].count("+")) <= 2)
                successp2_2 = successp_t1_2 and successp_t2_2
                # Evaluate rule 3
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

            # Add success for this model to overall number of successes
            successesp += successp
            successesp2 += successp2
            successesp3 += successp3
            successesp_2 += successp_2
            successesp2_2 += successp2_2
            successesp3_2 += successp3_2

        # Add success rates and equations to a single string for saving
        # IMPORTANT: Results from the pooled approach were supposed to be
        # marked. Due to a programming error, all equations were marked
        # as pooled. In reality, every eleventh time series was created
        # by the SINDy model provided with pooled time series while all
        # remaining ones were created by the model taking individual
        # time series.
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

    # In case training failed in all instances of training.
    except FileNotFoundError:

        string = "0;0;0;0;0;0;0"
        string_2 = "0;0;0;0;0;0;0"

    # Save results
    with open(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/SINDy_results.txt","w") as variable_name:
        variable_name.write(string)
    with open(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/SINDy_results_0.5.txt","w") as variable_name:
        variable_name.write(string_2)

if __name__ == "__main__":
    main()
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
import re
import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)

def _get_data(ts, theta, max, min, noisiness, *, key):
    """
    Returns a single time series (evaluated at the time points defined by ts) of the 
    Lotka-Volterra model with some normally-distributed noise. Initial conditions for 
    prey and predator are both chosen randomly from the range [min, max].

    Parameters
    ----------
    ts : jax.ArrayImpl
        An array containing all the time points the timeseries should be evaluated for.
    theta : list
        A list of four floats representing the parameters of the Lotka Volterra model
        [alpha, beta, gamma, delta].
    max : float
        Maximum value for the initial prey and predator values (before adding noise).
    min : float
        Minimum value for the initial prey and predator values (before adding noise).
    noisiness : float
        Scale of the normal distribution the noise is drawn from. If noisiness == 0,
        no noise is added.
    key : jax.ArrayImpl, optional
        A key used to make stochastic processes (in this case the noise values drawn 
        from a normal distribution) reproducible. If no key is provided, noise may
        differ between runs.

    Returns:
    --------
    jax.ArrayImpl
        An array containing a noisy Lotka Volterra time series, evaluated at time
        points ts.
    """
    
    y0 = jr.uniform(key, (2,), minval=min, maxval=max)

    def f(t, y, args):
        dXdt = theta[0] * y[0] - theta[1] * y[0] * y[1]
        dYdt = theta[2] * y[0] * y[1] - theta[3] * y[1]
        return jnp.stack([dXdt, dYdt], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    noise = jr.normal(key=key, shape=(len(ts), 2))
    ys += noisiness * noise
    return jnp.greater(ys, jnp.zeros(ys.shape)) * ys + 1e-8

def get_data(dataset_size, theta, max, min, t_end, datapoints, noisiness, *, key):
    """
    Returns multiple time series (evaluated at the time points defined by ts) of the 
    Lotka-Volterra model with some normally-distributed noise and different initial 
    conditions for prey and predator chosen randomly from the range [min, max].

    Parameters
    ----------
    dataset_size : int
        The amount of generated time series.
    theta : list
        A list of four floats representing the parameters of the Lotka Volterra model
        [alpha, beta, gamma, delta].
    max : float
        Maximum value for the initial prey and predator values (before adding noise).
    min : float
        Minimum value for the initial prey and predator values (before adding noise).
    t_end : float
        The last point in time that the time series are supposed to contain.
    datapoints : int
        The amount of evenly-spaced datapoints each time series is supposed to contain.
    noisiness : float
        Scale of the normal distribution the noise is drawn from. If noisiness == 0,
        no noise is added.
    key : jax.Array
        A key used to make stochastic processes (in this case the noise values drawn 
        from a normal distribution) reproducible. If no key is provided, noise may
        differ between runs.

    Returns:
    --------
    jax.ArrayImpl
        An array containing multiple noisy Lotka Volterra time series, evaluated at time
        points ts.
    """

    ts = jnp.linspace(0, t_end, datapoints)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, theta, max, min, noisiness, key=key))(key)
    return ts, ys

@click.command()
@click.option("-length", "--length_strategy", type=(float, float, float, float), default=(0.1, 1, -1, -1))
@click.option("-lr", "--lr_strategy", type=float, default=1e-3)
@click.option("-clip", "--clip_strategy", type=float, default=0.1)
@click.option("-batch", "--batch_size", type=int, default=20)
@click.option("-points", "--data_points", type=int, default=51)
@click.option("-noise", "--data_noise", type=float, default=0.0)
def main(length_strategy, lr_strategy, clip_strategy, batch_size, data_points, data_noise):

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

        # Retrieve observation data and real solution
        ys = jnp.stack(jnp.array([sim.observations.prey.values, sim.observations.predator.values]), axis=1)
        ts = sim.coordinates["time"]
        ts_obs = sim.observations.time.values
        data_vars = [x for x in sim.observations.data_vars]
        ts_real_sol, ys_real_sol = get_data(50, [1.3,0.9,0.8,1.8], 5, 1, 50, 1001, 0, key=jr.PRNGKey(0))

        # Create models from inference data
        models = [Func({"alpha":jnp.array(1.3), "delta":idata.posterior.sel(draw=m).delta.values[0]},weights=idata.posterior.sel(draw=m).weights.values.tolist()[0],bias=idata.posterior.sel(draw=m).bias.values.tolist()[0],key=model_key) for m in idata.posterior.draw.values]
        
        # Create simulated time series for all models
        res = [[evaluator._solver.standalone_solver(model, ts, y[:,0], ()) for y in ys] for model in models]

        def loss(y_obs, y_pred):
            return (y_obs - y_pred)**2

        def loss_func(y_obs, y_pred):
            return loss(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
        # Calculate losses for every model and determine best model 
        every = (int((1001 / data_points) / 10) + 1) * 10
        losses = [jnp.mean(loss_func(ys, jnp.array(result)[:,:,::every])) for result in res]
        best_model = sorted(zip(losses, [i for i in range(len(losses))]))[0][1]

        # For each validation time series, create figure containing data,
        # true solution, and predictions by all models
        for time_series in jnp.arange(0,10):

            fig, (ax_prey, ax_predator) = plt.subplots(nrows = 2, figsize = (7,9), sharex=True)

            ax_prey.plot(ts_real_sol, jnp.stack(ys_real_sol[40+time_series], axis=1)[0], ":", c="#404040", zorder=9)
            ax_predator.plot(ts_real_sol, jnp.stack(ys_real_sol[40+time_series], axis=1)[1], ":", c="#404040", zorder=9)
            ax_prey.plot(ts_obs, ys[time_series,0], "x", c="#404040", zorder=11)
            ax_predator.plot(ts_obs, ys[time_series,1], "x", c="#404040", zorder=11)

            for i in np.arange(len(models)):
                if i == best_model:
                    ax_prey.plot(ts, res[i][time_series][0], c="#007A9F", zorder=10)
                    ax_predator.plot(ts, res[i][time_series][1], c="#890000")
                else:
                    pass
                    ax_prey.plot(ts, res[i][time_series][0], c="#87E3FF", linewidth=1, alpha=0.4, zorder=0)
                    ax_predator.plot(ts, res[i][time_series][1], c="#FF4A4A", linewidth=1, alpha=0.2, zorder=0)

            ax_prey.set_ylim((0,8))
            ax_prey.set_ylabel("prey abundance", fontsize=12)
            ax_predator.set_ylim((0,8))
            ax_predator.set_xlabel("time", fontsize=12)
            ax_predator.set_ylabel("predator abundance", fontsize=12)

            fig.tight_layout()
            fig.savefig(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/validation_{str(time_series)}.png")

        # Create synthetic data for larger range of initial conditions
        ts_extrapol, ys_extrapol = get_data(10, [1.3,0.9,0.8,1.8], 10, 1, 50, 1001, 0, key=jr.PRNGKey(0))

        # Create simulated time series for all models
        res = [[evaluator._solver.standalone_solver(model, ts, y[0], ()) for y in ys_extrapol] for model in models]

        # For each extrapolation time series, create figure containing data,
        # true solution, and predictions by all models
        for time_series in jnp.arange(0,10):

            fig, (ax_prey, ax_predator) = plt.subplots(nrows = 2, figsize = (7,9), sharex=True)

            ax_prey.plot(ts_extrapol, jnp.stack(ys_extrapol[time_series], axis=1)[0], ":", c="#404040", zorder=9)
            ax_predator.plot(ts_extrapol, jnp.stack(ys_extrapol[time_series], axis=1)[1], ":", c="#404040", zorder=9)

            for i in np.arange(len(models)):
                if i == best_model:
                    ax_prey.plot(ts, res[i][time_series][0], c="#007A9F", zorder=10)
                    ax_predator.plot(ts, res[i][time_series][1], c="#890000")
                else:
                    pass
                    ax_prey.plot(ts, res[i][time_series][0], c="#87E3FF", linewidth=1, alpha=0.4, zorder=0)
                    ax_predator.plot(ts, res[i][time_series][1], c="#FF4A4A", linewidth=1, alpha=0.2, zorder=0)

            ax_prey.set_ylim((0,16))
            ax_prey.set_ylabel("prey abundance", fontsize=12)
            ax_predator.set_ylim((0,16))
            ax_predator.set_xlabel("time", fontsize=12)
            ax_predator.set_ylabel("predator abundance", fontsize=12)

            fig.tight_layout()
            fig.savefig(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/validation_extrapolation_{str(time_series)}.png")

        # Save index and loss of the best model
        with open(f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}_nfic/validation_results.txt","w") as variable_name:
            variable_name.write(str(best_model)+";"+str(losses[best_model]))

    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()
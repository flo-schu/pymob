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
        Scale of the normal distribution the noise is drawn from. I fnoisiness == 0,
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
@click.option("-lr", "--lr_strategy", type=(float, float, float, float), default=(3e-3, 3e-3, -1, -1))
@click.option("-clip", "--clip_strategy", type=(float, float, float, float), default=(0.1, 0.1, -1, -1))
@click.option("-batch", "--batch_size", type=int, default=20)
@click.option("-points", "--data_points", type=int, default=51)
@click.option("-noise", "--data_noise", type=float, default=0.0)
def main(length_strategy, lr_strategy, clip_strategy, batch_size, data_points, data_noise):

    sim = SimulationBase()
    sim.config.case_study.name = "lotka_volterra_UDE_case_study"
    sim.config.case_study.scenario = "UDETest"

    key = jrandom.PRNGKey(5678)
    data_key, model_key, loader_key = jrandom.split(key, 3)
    sim.model = Func({"alpha":jnp.array(1.3), "delta":jnp.array(1.8)},key=model_key)

    ts,ys = get_data(50, [1.3,0.9,0.8,1.8], 5, 0.1, 50, data_points, data_noise, key=jr.PRNGKey(0))
    datasets = jnp.linspace(0, 49, 50)
    test_data1 = xr.DataArray(ys[:,:,0], coords={"batch_id": datasets, "time": ts}).to_dataset(name="prey")
    test_data2 = xr.DataArray(ys[:,:,1], coords={"batch_id": datasets, "time": ts}).to_dataset(name="predator")
    test_data = xr.merge([test_data1, test_data2])
    sim.observations = test_data
    sim.model_parameters["y0"] = sim.observations.sel(time = 0).drop_vars("time")

    sim.config.model_parameters.alpha = Param(value=1.3, free=False)
    sim.config.model_parameters.delta = Param(value=1.8, free=True)
    sim.config.model_parameters.delta.prior = "uniform(loc=1.0,scale=2.0)"

    sim.solver = UDESolver
    sim.config.jaxsolver.max_steps = 100000
    sim.config.jaxsolver.throw_exception = False

    sim.dispatch_constructor()
    evaluator = sim.dispatch()

    sim.config.inference_optax.MLP_weight_dist = "normal()"
    sim.config.inference_optax.MLP_bias_dist = "normal()"
    sim.config.inference_optax.batch_size = batch_size
    sim.config.inference_optax.data_split = 0.8
    sim.config.inference_optax.multiple_runs_target = 10
    sim.config.inference_optax.multiple_runs_limit = 50

    sim.config.inference_optax.length_strategy = [i for i in length_strategy if i != -1]
    sim.config.inference_optax.steps_strategy = [1000 for i in length_strategy if i != -1]
    sim.config.inference_optax.lr_strategy = [i for i in lr_strategy if i != -1]
    sim.config.inference_optax.clip_strategy = [i for i in clip_strategy if i != -1]
    sim.set_inferer("optax")
    sim.inferer.run()

    sim.config.case_study.output_path = f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}"
    sim.config.case_study.data_path = f"hyperparams/scenario_{str(data_points)}_{str(data_noise)}_hyperparams_{str(length_strategy)}_{str(lr_strategy)}_{str(clip_strategy)}_{str(batch_size)}"
    sim.config.create_directory("results", force=True)
    os.makedirs(sim.data_path, exist_ok=True)
    os.makedirs(sim.output_path, exist_ok=True)

    sim.save_observations(force=True)
    sim.config.save(fp = sim.data_path+"/settings.cfg", force=True)
    try:
        sim.report()
    except AttributeError:
        pass
    sim.inferer.store_results()
    sim.inferer.store_loss_evolution()

if __name__ == "__main__":
    main()
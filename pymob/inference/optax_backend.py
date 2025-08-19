import jax
from pymob.inference.base import InferenceBackend, Distribution, Errorfunction
from typing import (
    Tuple, Dict, Union, Optional, Callable, Literal, List, Any,
    Protocol
)
import copy
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from functools import partial
import optax
import equinox as eqx
import xarray as xr

from case_studies.lotka_volterra_UDE_case_study.lotka_volterra_UDE_case_study.mod import Func

# I'd like to get rid of this but don't know how it works for the Param object. -> ask Flo
from pymob.sim.parameters import to_rv

scipy_to_jax = {
    # Continuous Distributions
    "beta": (jr.beta, {}),
    "cauchy": (jr.cauchy, {}),
    "chi2": (jr.chisquare, {}),
    "chisquare": (jr.chisquare, {}),
    "expon": (jr.exponential, {}),
    "exponential": (jr.exponential, {}),
    "gamma": (jr.gamma, {}),
    "gumbel": (jr.gumbel, {}),
    "laplace": (jr.laplace, {}),
    "logistic": (jr.logistic, {}),
    "lognorm": (jr.lognormal, {}),
    "lognormal": (jr.lognormal, {}),
    "norm": (jr.normal, {}),
    "normal": (jr.normal, {}),
    "pareto": (jr.pareto, {}),
    "rayleigh": (jr.rayleigh, {}),
    "uniform": (jr.uniform, {}),
    "t": (jr.t, {}),
    "triang": (jr.triangular, {}),
    "triangular": (jr.triangular, {}),
    "truncnorm": (jr.truncated_normal, {}),
    "truncnormal": (jr.truncated_normal, {}),
    "truncated_normal": (jr.truncated_normal, {}),
    "uniform": (jr.uniform, {}),
    "weibull_min": (jr.weibull_min, {}),
    "wald": (jr.wald, {}),
    
    # Discrete Distributions
    "bernoulli": (jr.bernoulli, {}),
    "binom": (jr.binomial, {}),
    "binomial": (jr.binomial, {}),
    "geom": (jr.geometric, {}),
    "geometric": (jr.geometric, {}),
    "poisson": (jr.poisson, {}),
    "randint": (jr.randint, {}),

    # some are missing, see https://docs.jax.dev/en/latest/jax.random.html for complete list -> TODO
}

def transform_observations(observations):
        
    ts = jnp.array(observations.time.values)
    data_vars = [x for x in observations.data_vars]
    if len(data_vars) > 1:
        ys_unstacked = jnp.array([y.values for (x,y) in observations.items()])
        ys = jnp.stack(ys_unstacked, axis=(len(ys_unstacked.shape)-1)) # check with Flo if this is a universal solution or if the stacked axis has to be adapted to the input data -> TODO
    else:
        ys = jnp.array([[y.values for (x,y) in observations.items()]])

    return ts, ys, data_vars

def transform_observations_backwards(ts, ys, data_vars):

    datasets = jnp.arange(ys.shape[0]) + 1

    return xr.Dataset({var: xr.DataArray(ys[:,:,i], coords={"batch_id": datasets, "time": ts}) for var, i in zip(data_vars, range(len(data_vars)))})

class OptaxDistribution(Distribution):
    distribution_map: Dict[str,Tuple[Callable, Dict[str,str]]] = scipy_to_jax
    parameter_converter = staticmethod(lambda x: jnp.array(x))

    def _get_distribution(self, distribution: str) -> Tuple[Callable, Dict[str, str]]:
        # TODO: This is not satisfying. I think the transformed distributions
        # should only be used when this is explicitly specified.
        # I really wonder, why this makes such a large change in numpyro
        return self.distribution_map[distribution]

    @property
    def dist_name(self):
        return self.distribution.func.__name__

class OptaxBackend(InferenceBackend):
    _distribution = OptaxDistribution
    prior: Dict[str, Callable]

    def __init__(self, simulation):
        super().__init__(simulation)

        if simulation.config.simulation.batch_dimension in [x for x in simulation.observations.sizes.keys()]:
            no_datasets = simulation.observations.sizes[simulation.config.simulation.batch_dimension]
        else:
            no_datasets = 1

        if no_datasets < self.config.inference_optax.batch_size:
            raise ValueError(f"The specified training batch size ({self.config.inference_optax.batch_size}) is larger than the number of batches in the data ({no_datasets}).")

    class StopOptimizing(Exception):
        pass

    def parse_deterministic_model(self):
        pass

    def parse_probabilistic_model(self):
        pass

    def posterior_predictions(self):
        pass

    def prior_predictions(self):
        pass

    def create_log_likelihood(self) -> Tuple[Errorfunction,Errorfunction]:
        # TODO: define
        return 

    def construct_model(self, reference_model):

        cfg = self.config.inference_optax
        params = {}

        for key in cfg.UDE_parameters.fixed:
            params[key] = (jnp.array(cfg.UDE_parameters[key].value), False)

        for key in cfg.UDE_parameters.free:
            dist = OptaxBackend._distribution(
                name=key, 
                random_variable=cfg.UDE_parameters[key].prior,
                dims=(),
                shape=()
            )

            sample = dist.construct(context=None, extra_kwargs={"key": jr.PRNGKey(np.random.randint(0,10000,()))})
            params[key] = (sample, True)

        dist = OptaxBackend._distribution(
            name="weights", 
            random_variable=to_rv(cfg.MLP_weight_dist),
            dims=(),
            shape=()
        )

        mlp_size = (reference_model.mlp.in_size, reference_model.mlp.out_size, reference_model.mlp.width_size, reference_model.mlp.depth)

        weights = dist.construct(context=None, extra_kwargs={"shape": (mlp_size[0]*mlp_size[2] + (mlp_size[3] - 1)*mlp_size[2]**2 + mlp_size[2]*mlp_size[1]), "key": jr.PRNGKey(np.random.randint(0,10000,()))})

        dist = OptaxBackend._distribution(
            name="bias", 
            random_variable=to_rv(cfg.MLP_bias_dist),
            dims=(),
            shape=()
        )

        bias = dist.construct(context=None, extra_kwargs={"shape": (mlp_size[3]*mlp_size[2] + mlp_size[1]), "key": jr.PRNGKey(np.random.randint(0,10000,()))})

        model_type = type(reference_model)

        return model_type(params, weights, bias, key=jr.PRNGKey(0))
    
    def optimize_model(self, model, observations):

        cfg = self.config.inference_optax

        # transform observations to suitable format

        ts = jnp.array(observations["time"].values)

        length_size = len(ts)

        # optimize model

        class StopOptimizing(Exception):
            pass

        def dataloader(observations, cfg, *, key):  
            dataset_size = observations.sizes[cfg.simulation.batch_dimension]
            observations = observations.rename({cfg.simulation.batch_dimension: "batch_id"})
            indices = jnp.arange(dataset_size)
            while True:
                perm = jr.permutation(key, indices)
                (key,) = jr.split(key, 1)
                start = 0
                end = cfg.inference_optax.batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    yield observations.isel(batch_id = batch_perm)
                    start = end
                    end = start + cfg.inference_optax.batch_size

        @eqx.filter_value_and_grad
        def grad_loss(model, ti, yi):
            sim_temp = copy.deepcopy(self.simulation)
            sim_temp.model = model
            sim_temp.observations = yi
            sim_temp.model_parameters["y0"] = sim_temp.observations.sel(time = 0).drop_vars("time") # first time value might not be zero

            evaluator_temp = sim_temp.dispatch()
            y_pred = transform_observations(evaluator_temp())[1]
            y_obs = transform_observations(yi)[1]

            # model_transform = lambda t, y: jnp.array(model(t,y))
            # y_pred = jax.vmap(model_transform, in_axes=(None, 0))(ti, yi[:, 0])
            return jnp.mean(cfg.loss_function(yi, y_pred))

        @eqx.filter_jit
        def make_step(ti, yi, model, opt_state):
            loss, grads = grad_loss(model, ti, yi)
            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        loader_key = jr.PRNGKey(np.random.randint(0,10000,()))

        for lr, steps, length, clip in zip(cfg.lr_strategy, cfg.steps_strategy, cfg.length_strategy, cfg.clip_strategy):

            sim_temp = copy.deepcopy(self.simulation)
            optim = optax.chain(optax.clip(clip), optax.adabelief(lr))
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            _ts = ts[: int(length_size * length)]
            _ys = observations.isel(time = slice(0, int(length_size * length)))

            if cfg.batch_size > 1: # FALSCH -> TODO
                for step, yi in zip(
                    range(steps), dataloader(_ys, self.config, key=loader_key)
                ):
                    print("Step " + str(step))
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state)

                    if not jnp.isfinite(loss).all():
                        raise StopOptimizing()

            else:
                for step, (yi,) in zip(
                    range(steps), [[_ys]] * steps
                ):
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state)

                    if not jnp.isfinite(loss).all():
                        raise StopOptimizing()

        return model
    
    def optimize_model2(self, model, observations):

        # transform observations to suitable format

        ts, ys, data_vars = transform_observations(observations)
        length_size = len(ts)

        # optimize model

        loss_func = self.config.inference_optax.loss_function
        transform_obs_back = lambda t, y: transform_observations_backwards(t, y, data_vars)

        loader_key = jr.PRNGKey(np.random.randint(0,10000,()))

        class StopOptimizing(Exception):
            pass

        def dataloader(arrays, batch_size, *, key):
            dataset_size = arrays[0].shape[0]
            assert all(array.shape[0] == dataset_size for array in arrays)
            indices = jnp.arange(dataset_size)
            while True:
                perm = jr.permutation(key, indices)
                (key,) = jr.split(key, 1)
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    yield tuple(array[batch_perm] for array in arrays)
                    start = end
                    end = start + batch_size

        @eqx.filter_value_and_grad
        def grad_loss(model, ti, yi, loss_func):
            y_obs = transform_obs_back(ti,yi)
            sim_temp = copy.deepcopy(self.simulation)
            sim_temp.model = model
            sim_temp.observations = y_obs
            sim_temp.model_parameters["y0"] = sim_temp.observations.sel(time = 0).drop_vars("time") # first time value might not be zero

            evaluator_temp = sim_temp.dispatch()
            y_pred = transform_observations(evaluator_temp())[1]

            # model_transform = lambda t, y: jnp.array(model(t,y))
            # y_pred = jax.vmap(model_transform, in_axes=(None, 0))(ti, yi[:, 0])
            return jnp.mean(loss_func(yi, y_pred))

        @eqx.filter_jit
        def make_step(ti, yi, model, opt_state, loss_func):
            loss, grads = grad_loss(model, ti, yi, loss_func)
            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        for lr, steps, length, clip in zip(self.config.inference_optax.lr_strategy, self.config.inference_optax.steps_strategy, self.config.inference_optax.length_strategy, self.config.inference_optax.clip_strategy):
            optim = optax.chain(optax.clip(clip), optax.adabelief(lr))
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            _ts = ts[: int(length_size * length)]
            _ys = ys[:, : int(length_size * length)]

            if self.config.inference_optax.batch_size > 1: # FALSCH -> TODO
                for step, (yi,) in zip(
                    range(steps), dataloader((_ys,), self.config.inference_optax.batch_size, key=loader_key)
                ):
                    print("Step " + str(step))
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state, loss_func)

                    if not jnp.isfinite(loss).all():
                        raise StopOptimizing()

            else:
                for step, (yi,) in zip(
                    range(steps), [[_ys]] * steps
                ):
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state)

                    if not jnp.isfinite(loss).all():
                        raise StopOptimizing()

        return model
    
    def optimize_model2b(self, model, observations):

        # transform observations to suitable format

        ts, ys, data_vars = transform_observations(observations)
        length_size = len(ts)

        multiple_datasets = self.config.simulation.batch_dimension in [x for x in observations.dims.keys()]
        if not multiple_datasets:
            ys = jnp.expand_dims(ys,0)

        # optimize model
        
        loader_key = jr.PRNGKey(np.random.randint(0,10000,()))

        def loss_func(y_obs, y_pred):
            return self.config.inference_optax.loss_function(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
        class StopOptimizing(Exception):
            pass

        def dataloader(arrays, batch_size, *, key):
            dataset_size = arrays[0].shape[0]
            assert all(array.shape[0] == dataset_size for array in arrays)
            indices = jnp.arange(dataset_size)
            while True:
                perm = jr.permutation(key, indices)
                (key,) = jr.split(key, 1)
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    yield tuple(array[batch_perm] for array in arrays)
                    start = end
                    end = start + batch_size

        @eqx.filter_value_and_grad
        def grad_loss(model, ti, yi, loss_func):
            y_pred = jnp.array(jax.vmap(self.simulation.evaluator._solver.standalone_solver, in_axes=(None, None, 0))(model, ti, yi[:, 0]))

            if y_pred.shape[0] > 1:
                y_pred = jnp.stack(y_pred, axis = (len(y_pred.shape)-1))

            losses = loss_func(yi, y_pred)
            return jnp.mean(losses)

        @eqx.filter_jit
        def make_step(ti, yi, model, opt_state, loss_func):
            loss, grads = grad_loss(model, ti, yi, loss_func)
            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            # jax.debug.breakpoint()
            return loss, model, opt_state

        for lr, steps, length, clip in zip(self.config.inference_optax.lr_strategy, self.config.inference_optax.steps_strategy, self.config.inference_optax.length_strategy, self.config.inference_optax.clip_strategy):
            optim = optax.chain(optax.clip(clip), optax.adabelief(lr))
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            _ts = ts[: int(length_size * length)]
            _ys = ys[:, : int(length_size * length)]

            if multiple_datasets:
                for step, (yi,) in zip(
                    range(steps), dataloader((_ys,), self.config.inference_optax.batch_size, key=loader_key)
                ):
                    print("Step " + str(step))
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state, loss_func)
                    print("Loss: " + str(loss))
                    if not jnp.isfinite(loss).all():
                        raise StopOptimizing()

            else:
                for step, (yi,) in zip(
                    range(steps), [[_ys]] * steps
                ):
                    print("Step " + str(step))
                    loss, model, opt_state = make_step(_ts, yi, model, opt_state, loss_func)
                    print("Loss: " + str(loss))
                    if not jnp.isfinite(loss).all():
                        raise StopOptimizing()

        return model
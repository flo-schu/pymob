import jax
from pymob.inference.base import InferenceBackend, Distribution, Errorfunction
from typing import (
    Tuple, Dict, Union, Optional, Callable, Literal, List, Any,
    Protocol
)
import copy
import warnings
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from functools import partial
import optax
import equinox as eqx
import xarray as xr
from tqdm import tqdm, TqdmWarning

from case_studies.lotka_volterra_UDE_case_study.lotka_volterra_UDE_case_study.mod import Func

# I'd like to get rid of this but don't know how it works for the Param object. -> ask Flo
from pymob.sim.parameters import to_rv

scipy_to_jax = {
    # Continuous Distributions
    "beta": (lambda a, b, key, loc=0.0, scale=1.0, shape=(): jr.beta(key=key, a=a, b=b, shape=shape)*scale + loc, {}),
    "cauchy": (lambda key, loc=0.0, scale=1.0, shape=(): jr.cauchy(key=key, shape=shape)*scale + loc, {}),
    "chi2": (lambda df, key, loc=0.0, scale=1.0, shape=(): jr.chisquare(key=key, df=df, shape=shape)*scale + loc, {}),
    "expon": (lambda key, loc=0.0, scale=1.0, shape=(): jr.exponential(key=key, shape=shape)*scale + loc, {}),
    "exponential": (lambda key, loc=0.0, scale=1.0, shape=(): jr.exponential(key=key, shape=shape)*scale + loc, {}),
    "gamma": (lambda a, key, loc=0.0, scale=1.0, shape=(): jr.gamma(key=key, a=a, shape=shape)*scale + loc, {}),
    "gumbel_r": (lambda key, loc=0.0, scale=1.0, shape=(): jr.gumbel(key=key, shape=shape)*scale + loc, {}),
    "laplace": (lambda key, loc=0.0, scale=1.0, shape=(): jr.laplace(key=key, shape=shape)*scale + loc, {}),
    "logistic": (lambda key, loc=0.0, scale=1.0, shape=(): jr.logistic(key=key, shape=shape)*scale + loc, {}),
    "lognorm": (lambda key, loc=0.0, scale=1.0, shape=(): jr.lognormal(key=key, shape=shape)*scale + loc, {}),
    "lognormal": (lambda key, loc=0.0, scale=1.0, shape=(): jr.lognormal(key=key, shape=shape)*scale + loc, {}),
    "norm": (lambda key, loc=0.0, scale=1.0, shape=(): jr.normal(key=key, shape=shape)*scale + loc, {}),
    "normal": (lambda key, loc=0.0, scale=1.0, shape=(): jr.normal(key=key, shape=shape)*scale + loc, {}),
    "pareto": (lambda b, key, loc=0.0, scale=1.0, shape=(): jr.pareto(key=key, b=b, shape=shape)*scale + loc, {}),
    "rayleigh": (lambda key, loc=0.0, scale=1.0, shape=(): jr.rayleigh(key=key, scale=1, shape=shape)*scale + loc, {}),
    "t": (lambda df, key, loc=0.0, scale=1.0, shape=(): jr.t(key=key, df=df, shape=shape)*scale + loc, {}),
    "triang": (lambda c, key, loc=0.0, scale=1.0, shape=(): jr.triangular(key=key, left=loc, mode=(loc+c*scale), right=(loc+scale), shape=shape), {}),
    "truncnorm": (lambda a, b, key, loc=0.0, scale=1.0, shape=(): jr.truncated_normal(key=key, lower=(a*scale+loc), upper=(b*scale+loc), shape=shape), {}),
    "truncnormal": (lambda a, b, key, loc=0.0, scale=1.0, shape=(): jr.truncated_normal(key=key, lower=(a*scale+loc), upper=(b*scale+loc), shape=shape), {}),
    "uniform": (lambda key, loc=0.0, scale=1.0, shape=(): jr.uniform(key=key, minval=loc, maxval=(loc+scale), shape=shape), {}),
    "wald": (lambda key, loc=0.0, scale=1.0, shape=(): jr.wald(key=key, mean=1, shape=shape)*scale + loc, {}),
    "weibull_min": (lambda c, key, loc=0.0, scale=1.0, shape=(): jr.weibull_min(key=key, scale=1, concentration=c, shape=shape)*scale + loc, {}),
    
    # Discrete Distributions
    "bernoulli": (lambda p, key, loc=0.0, shape=(): jr.bernoulli(key=key, p=p, shape=shape) + loc, {}),
    "binom": (lambda n, p, key, loc=0.0, shape=(): jr.binomial(key=key, n=n, p=p, shape=shape) + loc, {}),
    "geom": (lambda p, key, loc=0.0, shape=(): jr.geometric(key=key, p=p, shape=shape) + loc, {}),
    "poisson": (lambda mu, key, loc=0.0, shape=(): jr.poisson(key=key, lam=mu, shape=shape) + loc, {}),
    "randint": (lambda low, high, key, loc=0.0, shape=(): jr.randint(key=key, minval=low, maxval=high, shape=shape) + loc, {}),

    # some are missing, see https://docs.jax.dev/en/latest/jax.random.html for complete list -> TODO
}

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

    optimized_models: list

    def __init__(self, simulation):
        super().__init__(simulation)

        if simulation.config.simulation.batch_dimension in [x for x in simulation.observations.sizes.keys()]:
            self.n_datasets = simulation.observations.sizes[simulation.config.simulation.batch_dimension]
            self.n_train_sets = jnp.round(self.n_datasets * simulation.config.inference_optax.data_split).astype(int)
            if self.n_train_sets == self.n_datasets:
                self.n_train_sets -= 1
            if self.n_train_sets == 0:
                self.n_train_sets = 1
        else:
            self.n_datasets = 1
            self.n_train_sets = 1
            warnings.warn(
                "The single provided data batch will be used for both training and validation. " \
                "This should not be the case, please provide multiple datasets.",
                category=UserWarning
            )

        self.batch_size = self.config.inference_optax.batch_size

        if self.n_train_sets < self.batch_size:
            self.batch_size = self.n_train_sets
            warnings.warn(
                f"The specified training batch size ({self.config.inference_optax.batch_size}) is larger " \
                f"than the number of batches made available for training ({self.n_datasets}). The batch size " \
                f"was therefore lowered to {self.batch_size} (internally, the value in the config " \
                "stays the same).",
                category=UserWarning
            )

        if simulation.config.inference_optax.multiple_runs_target > simulation.config.inference_optax.multiple_runs_limit:
            self.multiple_runs_target = simulation.config.inference_optax.multiple_runs_limit
            warnings.warn(
                f"The specified target number for successful runs/output models ({simulation.config.inference_optax.multiple_runs_target}) " \
                f"is larger than the allowed total number of runs ({simulation.config.inference_optax.multiple_runs_limit}). " \
                f"The target was therefore lowered to {self.multiple_runs_target} (internally, the value in the config " \
                "stays the same).",
                category=UserWarning
            )
        else:
            self.multiple_runs_target = simulation.config.inference_optax.multiple_runs_target
        
        
    def run(self):
        models, success = self.optimize_multiple_runs()
        losses = [self.global_loss(model) for model in models]

        i = 0
        print("\nrun number\tsuccessful?\tloss\n")
        for (j, s) in enumerate(success):
            if s:
                print(f"run {j+1}\t\tyes\t\t{losses[i]}")
                i += 1
            else:
                print(f"run {j+1}\t\tno\t\t---")

        self.optimized_models = self.sort_models_by_global_loss(models)

    class StopOptimizing(Exception):
        pass

    def parse_deterministic_model(self):
        pass

    def parse_probabilistic_model(self):
        raise NotImplementedError("This method is currently not available for the Optax backend.")

    def posterior_predictions(self):
        raise NotImplementedError("This method is currently not available for the Optax backend.")

    def prior_predictions(self):
        raise NotImplementedError("This method is currently not available for the Optax backend.")

    def create_log_likelihood(self) -> Tuple[Errorfunction,Errorfunction]:
        raise NotImplementedError("This method is currently not available for the Optax backend.")
    
    def plot_likelihood_landscape(self, parameters, log_likelihood_func, gradient_func = None, bounds = ..., n_grid_points = 100, n_vector_points = 50, normal_base=False, ax = None):
        raise NotImplementedError("This method is currently not available for the Optax backend.")
    
    def plot_prior_predictions(self, data_variable, x_dim, ax=None, subset=..., n=None, seed=None, plot_preds_without_obs=False, prediction_data_variable = None, **plot_kwargs):
        raise NotImplementedError("This method is currently not available for the Optax backend.")
    
    def plot_posterior_predictions(self, data_variable, x_dim, ax=None, subset=..., n=None, seed=None, plot_preds_without_obs=False, prediction_data_variable = None, **plot_kwargs):
        raise NotImplementedError("This method is currently not available for the Optax backend.")
    
    def plot(self):
        raise NotImplementedError("This method is currently not available for the Optax backend.")
    
    def plot_diagnostics(self):
        raise NotImplementedError("This method is currently not available for the Optax backend.")
        
    def transform_observations(self, observations):
        ts = jnp.array(observations.time.values)
        data_vars = [x for x in observations.data_vars]
        ys_unstacked = jnp.array([y.values for (x,y) in observations.items()])
        ys = jnp.stack(ys_unstacked, axis=(len(ys_unstacked.shape)-1)) # check with Flo if this is a universal solution or if the stacked axis has to be adapted to the input data -> TODO

        return ts, ys, data_vars
    
    def transform_x_in(self, x_in):
        ts = jnp.array(x_in.time.values)
        ys = jnp.array([y.values for (x,y) in x_in.items()])

        return ts, ys

    def transform_observations_backwards(self, ts, ys, data_vars):
        datasets = jnp.arange(ys.shape[0]) + 1
        return xr.Dataset({var: xr.DataArray(ys[:,:,i], coords={"batch_id": datasets, "time": ts}) for var, i in zip(data_vars, range(len(data_vars)))})

    class StopOptimizing(Exception):
        pass

    def construct_model(self):
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
            random_variable=cfg.MLP_weight_dist,
            dims=(),
            shape=()
        )

        reference_model = self.simulation.model
        mlp_size = (reference_model.mlp.in_size, reference_model.mlp.out_size, reference_model.mlp.width_size, reference_model.mlp.depth)

        weights = dist.construct(context=None, extra_kwargs={"shape": (mlp_size[0]*mlp_size[2] + (mlp_size[3] - 1)*mlp_size[2]**2 + mlp_size[2]*mlp_size[1]), "key": jr.PRNGKey(np.random.randint(0,10000,()))})

        dist = OptaxBackend._distribution(
            name="bias", 
            random_variable=cfg.MLP_bias_dist,
            dims=(),
            shape=()
        )

        bias = dist.construct(context=None, extra_kwargs={"shape": (mlp_size[3]*mlp_size[2] + mlp_size[1]), "key": jr.PRNGKey(np.random.randint(0,10000,()))})

        model_type = type(reference_model)

        return model_type(params, weights, bias, key=jr.PRNGKey(0))
    
    def optimize_model(self, model, pbar):
        # transform observations to suitable format
        ts, ys, data_vars = self.transform_observations(self.simulation.observations)
        if self.n_datasets > 1:
            ys = ys[:self.n_train_sets]
        length_size = len(ts)

        if "x_in" in self.simulation.model_parameters.keys() and [x for x in self.simulation.model_parameters["x_in"].data_vars] != []:
            x_in_temp = self.transform_x_in(self.simulation.model_parameters["x_in"])
            x_in = (x_in_temp[0], x_in_temp[1][0])
        else:
            x_in = None

        multiple_datasets = self.config.simulation.batch_dimension in [x for x in self.simulation.observations.sizes.keys()]
        if not multiple_datasets:
            ys = jnp.expand_dims(ys,0)

        # optimize model
        loader_key = jr.PRNGKey(np.random.randint(0,10000,()))

        def loss_func(y_obs, y_pred):
            return self.config.inference_optax.loss_function(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
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
        def grad_loss(model, ti, yi, x_in, loss_func):
            y_pred = jnp.array(jax.vmap(self.simulation.evaluator._solver.standalone_solver, in_axes=(None, None, 0, None))(model, ti, yi[:, 0], x_in))
            y_pred = jnp.stack(y_pred, axis = (len(y_pred.shape)-1))

            losses = loss_func(yi, y_pred)
            return jnp.mean(losses)

        @eqx.filter_jit
        def make_step(ti, yi, x_in, model, opt_state, loss_func):
            loss, grads = grad_loss(model, ti, yi, x_in, loss_func)
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
                    loss, model, opt_state = make_step(_ts, yi, x_in, model, opt_state, loss_func)
                    pbar.update(length)
                    if not jnp.isfinite(loss).all():
                        raise self.StopOptimizing()

            else:
                for step, (yi,) in zip(
                    range(steps), [[_ys]] * steps
                ):
                    loss, model, opt_state = make_step(_ts, yi, x_in, model, opt_state, loss_func)
                    pbar.update(length)
                    if not jnp.isfinite(loss).all():
                        raise self.StopOptimizing()

        return model
    
    def optimize_multiple_runs(self):
        cfg = self.config.inference_optax

        tried_runs = successful_runs = 0

        models = []
        success = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmWarning)

            pbar = tqdm(total = self.multiple_runs_target * jnp.sum(jnp.array(cfg.steps_strategy) * jnp.array(cfg.length_strategy)).item(), desc=f"{successful_runs} of {self.multiple_runs_target} runs completed")

            while tried_runs < cfg.multiple_runs_limit and successful_runs < self.multiple_runs_target:

                runstr = "run" if (tried_runs-successful_runs)==1 else "runs"
                pbar.set_postfix_str(f"{tried_runs - successful_runs} unsuccessful {runstr} so far")
                tried_runs += 1
                
                try:

                    optimizable_model = self.construct_model()
                    optimized_model = self.optimize_model(optimizable_model, pbar)

                    models.append(optimized_model)
                    successful_runs += 1
                    pbar.set_description(f"{successful_runs} of {self.multiple_runs_target} runs completed")
                    success.append(True)

                except self.StopOptimizing:

                    success.append(False)
                    pbar.n = successful_runs * jnp.sum(jnp.array(cfg.steps_strategy) * jnp.array(cfg.length_strategy)).item()
                    pbar.last_print_n = successful_runs * jnp.sum(jnp.array(cfg.steps_strategy) * jnp.array(cfg.length_strategy)).item()
                    pass

        if successful_runs < self.multiple_runs_target:
            warnings.warn(
                "Target number of successful runs was not reached before surpassing the " \
                f"allowed total number of runs. Only {successful_runs} optimized models were returned."
            )

        return models, success
    
    def global_loss(self, model):
        ts, ys, data_vars = self.transform_observations(self.simulation.observations)
        if self.n_datasets > 1:
            ys = ys[self.n_train_sets:]
        else:
            ys = jnp.array([ys,])

        if "x_in" in self.simulation.model_parameters.keys() and [x for x in self.simulation.model_parameters["x_in"].data_vars] != []:
            x_in_temp = self.transform_x_in(self.simulation.model_parameters["x_in"])
            x_in = (x_in_temp[0], x_in_temp[1][0])
        else:
            x_in = None

        def loss_func(y_obs, y_pred):
            return self.config.inference_optax.loss_function(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
        @eqx.filter_jit
        def loss(model, ti, yi, loss_func):
            y_pred = jnp.array(jax.vmap(self.simulation.evaluator._solver.standalone_solver, in_axes=(None, None, 0, None))(model, ti, yi[:, 0], x_in))
            y_pred = jnp.stack(y_pred, axis = (len(y_pred.shape)-1))

            losses = loss_func(yi, y_pred)
            return jnp.mean(losses)
        
        return loss(model, ts, ys, loss_func)
    
    def sort_models_by_global_loss(self, models):
        losses = [self.global_loss(model) for model in models]

        sorted_models = []
        sorted_losses = []

        for x, y in sorted(zip(losses, models)):
            sorted_models.append(y)
            sorted_losses.append(x)

        return sorted_models
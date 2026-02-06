import time
import jax
from pymob.inference.base import InferenceBackend, Distribution, Errorfunction
from pymob.utils.UDE import getFuncBias, transformBias, getFuncWeights, transformWeights
from typing import (
    Tuple, Dict, Union, Optional, Callable, Literal, List, Any,
    Protocol
)
from pymob.sim.parameters import optim_lib
import copy
import warnings
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from functools import partial
import optax
import equinox as eqx
from equinox import EquinoxRuntimeError
import xarray as xr
import arviz as az
from tqdm import tqdm, TqdmWarning
import matplotlib.pyplot as plt


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

# optim_lib = {
#     "adabelief": lambda parameters: optax.adabelief(**parameters),
#     "adadelta": lambda parameters: optax.adadelta(**parameters),
#     "adan": lambda parameters: optax.adan(**parameters),
#     "adafactor": lambda parameters: optax.adafactor(**parameters),
#     "adagrad": lambda parameters: optax.adagrad(**parameters),
#     "adam": lambda parameters: optax.adam(**parameters),
#     "adamw": lambda parameters: optax.adamw(**parameters),
#     "adamax": lambda parameters: optax.adamax(**parameters),
#     "adamaxw": lambda parameters: optax.adamaxw(**parameters),
#     "amsgrad": lambda parameters: optax.amsgrad(**parameters),
#     "fromage": lambda parameters: optax.fromage(**parameters),
#     "lamb": lambda parameters: optax.lamb(**parameters),
#     "lars": lambda parameters: optax.lars(**parameters),
#     "lbfgs": lambda parameters: optax.lbfgs(**parameters), # problematic, only works with argument linesearch=None due to dimensionless mechanistic parameters
#     "lion": lambda parameters: optax.lion(**parameters),
#     "nadam": lambda parameters: optax.nadam(**parameters),
#     "nadamw": lambda parameters: optax.nadamw(**parameters),
#     "noisy_sgd": lambda parameters: optax.noisy_sgd(**parameters),
#     "novograd": lambda parameters: optax.novograd(**parameters),
#     "optimistic_gradient_descent": lambda parameters: optax.optimistic_gradient_descent(**parameters),
#     "optimistic_adam_v2": lambda parameters: optax.optimistic_adam_v2(**parameters),
#     "polyak_sgd": lambda parameters: optax.polyak_sgd(**parameters),
#     "radam": lambda parameters: optax.radam(**parameters),
#     "rmsprop": lambda parameters: optax.rmsprop(**parameters),
#     "rprop": lambda parameters: optax.rprop(**parameters),
#     "sgd": lambda parameters: optax.sgd(**parameters),
#     "sign_sgd": lambda parameters: optax.sign_sgd(**parameters),
#     "signum": lambda parameters: optax.signum(**parameters), # problematic, does not exist in the installed optax package despite being listed on the website
#     "sm3": lambda parameters: optax.sm3(**parameters), # problematic due to dimensionless mechanistic parameters
#     "yogi": lambda parameters: optax.yogi(**parameters),
# }

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

    '''
    A class used to generate and train models with the same properties as defined
    by the model inside the SimulationBase object.

    Attributes
    ----------
    _distribution : OptaxDistribution
        An instance of the OptaxDistribution class translating from SciPy to JAX
        distribution names.
    optimized_models : list
        A list containing all trained models. If the inference_optax.indepth
        configuration option is set to "full", this also contains all intermediary
        states from the successful instances of training.
    failed_models : list
        A list containing all models from failed instances of training. Empty if
        the inference_optax.indepth configuration option is set to "off". Contains 
        the model states from the final steps of training before failure if the
        inference_optax.indepth configuration option is set to "partial", and all
        intermediary states if it is set to "full".

    Methods
    -------
    run(self, return_losses = False, key=jr.PRNGKey(0))
        Main method of the OptaxInferer class. Calls all relevant methods for
        preparing the training procedure, training multiple models, creating
        inference data and, if the inference_optax.indepth configuration option
        is set to "partial" or "full", creating some insight about the training
        process.
    plot_posterior_predictions(
        self, data_variable: str, x_dim: str, ax=None, subset={}, n=1, seed=None, 
        plot_preds_without_obs=False, prediction_data_variable: Optional[str] = None
    )
        Plots the predictions of the optimized models against the data. The model
        with the lowest loss is plotted individually while the other models
        constitute a filled area between the maximum and minimum predicted values.
    transform_observations(self, observations)
        Transforms the given observations to a format that some other methods
        require.
    transform_x_in(self, x_in)
        Transforms the given input data to a format that some other methods
        require.
    compile_make_step(self, key)
        Returns a pre-compiled method running all the necessary actions of a single
        training step.
    construct_model(self, key)
        Returns a model with the same properties as the model stored in the
        SimulationBase object, except for the values of the trainable parameters
        which are drawn randomly from the distributions defined in the
        configuration settings.
    optimize_model(self, model, pbar, make_step, key)
        Takes a model and optimizes it with respect to the loss function defined
        by the model class. Returns the trained model along with some information
        about its training process, depending on the value of the
        inference_optax.indepth configuration option.
    optimize_multiple_runs(self, make_step, key)
        Constructs and trains multiple models until either the multiple_runs_target
        or the multiple_runs_limit ceiling defined in the configuration is reached.
    global_loss(self, model)
        Returns the loss value associated with the model passed to the function by
        comparing simulation results with the validation data. Computes its results
        by using the loss function defined by the model class.
    sort_models_by_global_loss(self, models)
        Returns a list of indices by calculating the loss values of each optimized 
        model with respect to the validation data and sorting the indices of the
        models by these losses in ascending order.
    create_idata(self)
        Creates and returns inference data for the optimized and, if the
        inference_optax.indepth configuration option is set to "partial" or "full"
        the last state of models from the failed instances of training.
    store_results(self, output=None, output_f=None)
        Stores inferences data. The location can be chosen by passing it to the
        function as an input parameter.
    load_results(self, file="optax_idata.nc", cluster: Optional[int] = None)
        Loads inference data from the specified location.
    store_results(self, output=None, output_f=None)
        Stores loss evolution data. The location can be chosen by passing it to the
        function as an input parameter.
    load_results(self, file="optax_idata.nc", cluster: Optional[int] = None)
        Loads loss evolution data from the specified location.
    '''


    _distribution = OptaxDistribution

    optimized_models: list
    failed_models: list

    def __init__(self, simulation):
        """
        Initializes the inferer. Splits data into training and validation data. Warns users
        if too little data is provided or if some config options are not defined in a
        meaningful way.
        
        Parameters
        ----------
        simulation : SimulationBase
            The simulation for which the inferer is created.
        """
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
        
    def run(self, key=jr.PRNGKey(0)):
        """
        Runs the inferer. First, the function that is run every step is JIT-compiled. Then,
        a number (as defined by the inference_optax.multiple_runs_target and
        inference_optax.multiple_runs_limit configuration options) of trained models is
        produced, and a summarizing message is being shown to the user. Lastly, inference data
        and, optionally, further data about the training process is generated.
        
        Parameters
        ----------
        key : jax.ArrayImpl
            A key fixing all stochastic processes during the multiple instances of training.

        """

        make_step_compiled = self.compile_make_step(key)
        self.optimized_models, self.failed_models, success, timeouts, lossev = self.optimize_multiple_runs(make_step_compiled, key)
        if self.config.inference_optax.indepth == "full":
            losses = [self.global_loss(model[-1]) for model in self.optimized_models]
        else:
            losses = [self.global_loss(model) for model in self.optimized_models]

        i = 0
        print("\nrun number\tsuccessful?\tloss\n")
        for (j, s) in enumerate(success):
            if s:
                print(f"run {j+1}\t\tyes\t\t{losses[i]}")
                i += 1
            else:
                print(f"run {j+1}\t\tno\t\t---")

        self.idata, self.idata_f = self.create_idata()  
        if self.config.inference_optax.indepth != "off":
            self.lossev = xr.DataArray(lossev, coords={"run": jnp.arange(1, lossev.shape[0]+1), "step": jnp.arange(1, lossev.shape[1]+1)}).to_dataset(name="losses")
            self.timeouts = timeouts
    
    def run2(self):
        raise NotImplementedError()
        make_step_compiled = self.compile_make_step2()
        self.optimized_models, self.failed_models, success, lossev = self.optimize_multiple_runs2()
        losses = [self.global_loss2(model) for model in self.optimized_models]

        i = 0
        print("\nrun number\tsuccessful?\tloss\n")
        for (j, s) in enumerate(success):
            if s:
                print(f"run {j+1}\t\tyes\t\t{losses[i]}")
                i += 1
            else:
                print(f"run {j+1}\t\tno\t\t---")

        self.idata = self.create_idata2()  
        self.lossev = xr.DataArray(lossev, coords={"run": jnp.arange(1, lossev.shape[0]+1), "step": jnp.arange(1, lossev.shape[1]+1)}).to_dataset(name="losses")
    
    @property
    def best_model(self):
        return self.optimized_models[0]

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
    
    def plot_posterior_predictions(
        self, data_variable: str, x_dim: str, ax=None, subset={},
        n=1, seed=None, plot_preds_without_obs=False,
        prediction_data_variable: Optional[str] = None
    ):
        """
        Graphically displays the predictions made by trained models. The data variable for
        which predictions should be plotted as well as the number of graphs to be plotted
        can be chosen by the user. Initial conditions are taken from the observations stored
        by the SimulationBase.

        Parameters
        ----------
        data_variable : str
            The name of the data variable for which observations should be plotted.
        x_dim : str
            The name of the dimension that should be shown on the x-axis.
        ax : matplotlib.axes._axes.Axes
            Axes on which the posterior predictions should be plotted. If this is set to None, 
            which is the default, new axes are being created.
        subset : dict
            Subset of the posterior predictions to be plotted. Has to be given in the format
            used by xArray's sel() method.
        n : int
            Number of plot that should be created. By default, this is set to 1.
        seed : Any
            Not used in this implementation.
        plot_preds_without_obs : boolean
            Determines whether observation data is omitted in the plotted graphic. By default,
            this is set to False, that is, observations are plotted.
        prediciton_data_variable : str, optional
            The name of the state variable for which predictions should be plotted. If this
            is not passed to the method, the variable defined by data_variable is plotted.

        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            The axes on which the posterior predictions were plotted.
        """
        observations = self.simulation.observations

        if self.n_datasets > 1:
            if n > (self.n_datasets - self.n_train_sets):
                msgstr = "dataset is" if (self.n_datasets - self.n_train_sets)==1 else "datasets are"
                warnings.warn(
                    f"The specified number of plotted datasets ({n}) is greater than " \
                    f"the number of validation datasets ({int(self.n_datasets - self.n_train_sets)}). " \
                    f"Therefore, only {int(self.n_datasets - self.n_train_sets)} {msgstr} being plotted.",
                    category = UserWarning
                )
                n = int(self.n_datasets - self.n_train_sets)
            observations = observations.isel({self.simulation.config.simulation.batch_dimension: slice(int(self.n_train_sets), int(self.n_train_sets + n))})
        else:
            if n > 1:
                warnings.warn(
                    f"The specified number of plotted datasets ({n}) is greater than " \
                    "the number of validation datasets (1). " \
                    "Therefore, only 1 dataset is being plotted.",
                    category = UserWarning
                )
                n = 1
            observations = observations.expand_dims(self.simulation.config.simulation.batch_dimension)
            observations = observations.assign_coords({self.simulation.config.simulation.batch_dimension:[0]})

        predictions = self.idata.posterior_model_fits.isel({"data_batch": slice(int(self.n_train_sets), int(self.n_train_sets + n))})

        # filter subset coordinates present in data_variable
        subset = {k: v for k, v in subset.items() if k in observations.coords}
        
        if prediction_data_variable is None:
            prediction_data_variable = data_variable

        # select subset
        if prediction_data_variable in predictions:
            preds = predictions.sel(subset)[prediction_data_variable]
        else:
            raise KeyError(
                f"{prediction_data_variable} was not found in the predictions "+
                "consider specifying the data variable for the predictions "+
                "explicitly with the option `prediction_data_variable`."
            )
        try:
            obs = observations.sel(subset)[data_variable]
        except KeyError:
            obs = preds.copy().mean(dim=("chain", "draw"))
            obs.values = np.full_like(obs.values, np.nan)

        best_model_index = [x for x, y in enumerate(self.sort_models_by_global_loss2(self.optimized_models)) if y==0][0]

        if ax is None:
            _, ax = plt.subplots(ncols=1, nrows=n, figsize=(5,3*n), constrained_layout = True)

        for j in jnp.arange(n):

            if n > 1:
                current_axis = ax[j]
            else:
                current_axis = ax

            maxima = jnp.array([jnp.max(preds.values[0,:,j][:,i]) for i in jnp.arange(preds.values[0,:,j].shape[1])])
            minima = jnp.array([jnp.min(preds.values[0,:,j][:,i]) for i in jnp.arange(preds.values[0,:,j].shape[1])])

            best_model_results = preds.values[0,best_model_index,j]

            if not plot_preds_without_obs:
                current_axis.plot(obs[x_dim].values, obs.values[j], "o", markersize=3, label="observations")
            current_axis.plot(obs[x_dim].values, best_model_results, c="grey", label="model with lowest loss")
            current_axis.fill_between(obs[x_dim].values, minima, maxima, color="lightgrey", label="range of all models")

            current_axis.set(xlabel = x_dim, ylabel = data_variable)    

        if n > 1:
            ax[0].legend()
        else:
            ax.legend()

        return ax
    
    def plot(self):
        raise NotImplementedError("This method is currently not available for the Optax backend.")
    
    def plot_diagnostics(self):
        raise NotImplementedError("This method is currently not available for the Optax backend.")
        
    def transform_observations(self, observations):
        """
        Transforms observations from the xArray to the jax.numpy format.

        Parameters
        ----------
        observations : xarray.core.dataset.Dataset
            An xArray dataset containing observation data.

        Returns
        -------
        ts : jax.ArrayImpl
            A jax.numpy array containing the coordinates along the time axis.
        ys : jax.ArrayImpl
            A jax.numpy array containing all the observation data.
        data_vars : list
            A list containing the names (as str) of all data variables.
        """
        ts = jnp.array(observations.time.values)
        data_vars = [x for x in observations.data_vars]
        ys_unstacked = jnp.array([y.values for (x,y) in observations.items()])
        ys = jnp.stack(ys_unstacked, axis=(len(ys_unstacked.shape)-1))

        return ts, ys, data_vars
    
    def transform_x_in(self, x_in):
        """
        Transforms input data from the xArray to the jax.numpy format.

        Parameters
        ----------
        x_in : xarray.core.dataset.Dataset
            An xArray dataset containing input data.

        Returns
        -------
        ts : jax.ArrayImpl
            A jax.numpy array containing the coordinates along the time axis.
        ys : jax.ArrayImpl
            A jax.numpy array containing all the input data.
        """
        ts = jnp.array(x_in.time.values)
        ys = jnp.array([y.values for (x,y) in x_in.items()])

        return ts, ys

    # def transform_observations_backwards(self, ts, ys, data_vars):
    #     datasets = jnp.arange(ys.shape[0]) + 1
    #     return xr.Dataset({var: xr.DataArray(ys[:,:,i], coords={"batch_id": datasets, "time": ts}) for var, i in zip(data_vars, range(len(data_vars)))})
    
    def compile_make_step(self, key):
        """
        JIT-compiles the make_step function used during every training step.

        Parameters
        ----------
        key : jax.ArrayImpl
            A key fixing all stochastic processes during training.

        Returns
        -------
        make_step_jit_compiled : equinox._compile_utils.Compiled
            Compiled version of the make_step function.
        """
        @eqx.filter_value_and_grad
        def grad_loss(model, ti, yi, mask, t_thresh, x_in, loss_func):
            """
            Computes the loss and gradients of all model parameters.

            Parameters
            ----------
            model : Any
                Model object containing all model parameters somewhere within a pytree
                structure and returning derivatives depending on the model states when
                called with the model states as input.
            ti : jax.ArrayImpl
                A jax.numpy array containing the coordinates along the time axis.
            yi : jax.ArrayImpl
                A jax.numpy array containing a batch of observation data.
            mask : jax.ArrayImpl
                An array containing True for all observations that should be considered
                during loss calculation, and False for all other observations.
            t_thresh : float
                The point in time up to which model predictions are calculated.
            x_in : jax.ArrayImpl
                A jax.numpy array containing a batch of input data.
            """
            y_pred = jnp.array(jax.vmap(self.simulation.evaluator._solver.standalone_solver, in_axes=(None, None, 0, None, None))(model, ti, yi[:, 0], x_in, t_thresh))
            y_pred = jnp.stack(y_pred, axis = (len(y_pred.shape)-1))

            losses = loss_func(yi, y_pred)

            return jnp.mean(jnp.where(mask, losses, mask))

        def make_step(ti, yi, x_in, mask, t_thresh, model, optim, opt_state, loss_func):
            """
            Executes all the necessary tasks for one step of training. Calculates the loss
            and gradients with respect to all model parameters, uses the gradients to
            compute an update to the model, and applies these updates.

            Parameters
            ----------
            ti : jax.ArrayImpl
                A jax.numpy array containing the coordinates along the time axis.
            yi : jax.ArrayImpl
                A jax.numpy array containing a batch of observation data.
            x_in : jax.ArrayImpl
                A jax.numpy array containing a batch of input data.
            mask : jax.ArrayImpl
                An array containing True for all observations that should be considered
                during loss calculation, and False for all other observations.
            t_thresh : float
                The point in time up to which model predictions are calculated.
            model : Any
                Model object containing all model parameters somewhere within a pytree
                structure and returning derivatives depending on the model states when
                called with the model states as input.
            optim : optax._src.base.GradientTransformationExtraArgs
                Optimizer object calculating model updates depending on the gradient and
                the state of the optimization.
            opt_state : tuple
                Current state of the optimization.
            loss_func : Callable
                Loss function calculating the error for a given pair of values (y_obs,
                y_pred), with y_obs being the observed and y_pred being the predicted
                value.

            Returns
            -------
            loss : jax.ArrayImpl
                A jax.numpy array containing the current loss as its only entry.
            model : Any
                Model object with updated parameters.
            opt_state : tuple
                New state of the optimization.
            """
            loss, grads = grad_loss(model, ti, yi, mask, t_thresh, x_in, loss_func)
            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array), value=loss, grad=grads, value_fn=grad_loss)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state
        
        def construct_optimizer(random_variable, clip):
            parameters = {key: entry.evaluate() for key, entry in random_variable.parameters.items()}
            optim = optim_lib[random_variable.distribution](parameters)
            if clip != 0:
                optim = optax.chain(optax.clip(clip), optim)
            return optim
        
        make_steps = []

        make_step_jit = eqx.filter_jit(make_step)

        ts, ys, _ = self.simulation.inferer.transform_observations(self.simulation.observations)

        if self.n_datasets == 1:
            ys = jnp.expand_dims(ys, 0)

        if "x_in" in self.simulation.model_parameters.keys() and [x for x in self.simulation.model_parameters["x_in"].data_vars] != []:
            x_in_temp = self.transform_x_in(self.simulation.model_parameters["x_in"])
            x_in = (x_in_temp[0], x_in_temp[1][0])
        else:
            x_in = None

        model = self.construct_model(key)

        def loss_func(y_obs, y_pred):
            return self.simulation.model.loss(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
        
        for optimizer, clip in zip(self.config.inference_optax.optim_strategy, self.config.inference_optax.clip_strategy):

            optim = construct_optimizer(optimizer, clip)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            
            make_steps.append(make_step_jit.lower(ts, ys[0:self.config.inference_optax.batch_size], x_in, jnp.ones(ys[0:self.config.inference_optax.batch_size].shape), ts[-1], model, optim, opt_state, loss_func).compile())

        return make_steps
    
    def compile_make_step2(self):
        raise NotImplementedError()
        @eqx.filter_value_and_grad
        def grad_loss(model, yi, batch, mask, evaluator, data_vars, loss_func):
            evaluator.model = model
            evaluator()
            y_pred = jnp.array([evaluator.Y[data_var] for data_var in data_vars])
            y_pred = jnp.stack(y_pred, axis = (len(y_pred.shape)-1))[:,:yi.shape[1]]

            losses = loss_func(yi[batch], y_pred[batch])
            return jnp.mean(jnp.where(mask, losses, mask))

        def make_step(yi, batch, mask, model, evaluator, data_vars, opt_state, loss_func):
            loss, grads = grad_loss(model, yi, batch, mask, evaluator, data_vars, loss_func)
            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        make_step_jit = eqx.filter_jit(make_step)

        ts, ys, data_vars = self.simulation.inferer.transform_observations(self.simulation.observations)

        if "x_in" in self.simulation.model_parameters.keys() and [x for x in self.simulation.model_parameters["x_in"].data_vars] != []:
            x_in_temp = self.transform_x_in(self.simulation.model_parameters["x_in"])
            x_in = (x_in_temp[0], x_in_temp[1][0])
        else:
            x_in = None

        model = self.construct_model()

        clip = self.config.inference_optax.clip_strategy
        lr = self.config.inference_optax.lr_strategy
        batch_size = self.config.inference_optax.batch_size
        evaluator = self.simulation.dispatch()

        if clip != 0:
            optim = optax.chain(optax.clip(clip), optax.adabelief(lr))
        else:
            optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        def loss_func(y_obs, y_pred):
            return self.simulation.model.loss(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
        
        return make_step_jit.lower(model, ys[0:batch_size], jnp.arange(batch_size), ys.shape[1], jnp.ones(ys[0:self.config.inference_optax.batch_size].shape), evaluator, data_vars, loss_func).compile()

    class StopOptimizing(Exception):
        pass

    def construct_model(self, key):
        """
        Constructs a new model with the same structure as the model stored within
        self.simulation but different parameter values. Parameter values are drawn from
        distributions specified by the user as priors.

        Parameters
        ----------
        key : jax.ArrayImpl
            A key fixing the stochastic processes during random parameter generation.
        """
        cfg = self.config
        params = {}

        for param in cfg.model_parameters.fixed:
            params[param] = (jnp.array(cfg.model_parameters[param].value), False)

        for param in cfg.model_parameters.free:
            dist = OptaxBackend._distribution(
                name=param, 
                random_variable=cfg.model_parameters[param].prior,
                dims=(),
                shape=()
            )

            sample = dist.construct(context=None, extra_kwargs={"key": key})
            params[param] = (sample, True)

        dist = OptaxBackend._distribution(
            name="weights", 
            random_variable=cfg.inference_optax.MLP_weight_dist,
            dims=(),
            shape=()
        )

        reference_model = self.simulation.model
        mlp_size = (reference_model.mlp.in_size, reference_model.mlp.out_size, reference_model.mlp.width_size, reference_model.mlp.depth)

        weights = dist.construct(context=None, extra_kwargs={"shape": (mlp_size[0]*mlp_size[2] + (mlp_size[3] - 1)*mlp_size[2]**2 + mlp_size[2]*mlp_size[1]), "key": key})

        dist = OptaxBackend._distribution(
            name="bias", 
            random_variable=cfg.inference_optax.MLP_bias_dist,
            dims=(),
            shape=()
        )

        bias = dist.construct(context=None, extra_kwargs={"shape": (mlp_size[3]*mlp_size[2] + mlp_size[1]), "key": key})

        model_type = type(reference_model)

        model = model_type(params, weights, bias, key=key)

        return model
    
    def optimize_model(self, model, pbar, make_step, key):
        """
        Optimizes a single model with respect to the observation data. Applies a stochastic
        gradient descent algorithm in order to arrive at a suitable parameterization.

        Parameters
        ----------
        model : Any
            Model object containing all model parameters somewhere within a pytree
            structure and returning derivatives depending on the model states when
            called with the model states as input.
        pbar : tqdm.std.tqdm
            Progress bar being updated at every step.
        make_step : equinox._compile_utils.Compiled
            Compiled version of the make_step function.
        key : jax.ArrayImpl
            A key fixing all stochastic processes during training.

        Returns
        -------
        model : Any or list[Any]
            Depending on the value of the configuration option inference_optax.indepth, this
            contains either the state of the model after the last step of training or after
            every step of training.
        success : bool
            Reports whether training was successful or not.
        timeout : bool
            Reports whether training has timed out after the amount of time specified by the
            inference_optax.timeout configuration option.
        lossev : list[jax.ArrayImpl]
            Depending on the value of the configuration option inference_optax.indepth, this
            returns either a None value or a list of loss values assumed over the course of 
            the training process.
        """
        start_time = time.time()
        # transform observations to suitable format
        ts, ys, data_vars = self.transform_observations(self.simulation.observations)
        if self.n_datasets > 1:
            ys = ys[:self.n_train_sets]
        else:
            ys = jnp.expand_dims(ys,0)
        length_size = len(ts)

        # transform input data to suitable format (if available)
        if "x_in" in self.simulation.model_parameters.keys() and [x for x in self.simulation.model_parameters["x_in"].data_vars] != []:
            x_in_temp = self.transform_x_in(self.simulation.model_parameters["x_in"])
            x_in = (x_in_temp[0], x_in_temp[1][0])
        else:
            x_in = None

        loader_key = key

        if self.config.inference_optax.indepth == "full":
            model_list = [model]
        if self.config.inference_optax.indepth != "off":
            lossev_single_model = []

        # wrap loss func so missing values from observations are filled in with predicted values
        # (results in loss=0 for these values)
        def loss_func(y_obs, y_pred):
            return self.simulation.model.loss(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
        def dataloader(arrays, batch_size, *, key):
            """
            Splits the data into smaller batches for training.

            Parameters
            ----------
            arrays : jax.ArrayImpl
                The full observation data set.
            batch_size : int
                The amount of time series a single batch is supposed to contain.
            key : jax.ArrayImpl
                A key fixing all stochastic processes during the generation of batches.

            Yields
            ------
            jax.ArrayImpl
                An array containing the spedified number of Lotka Volterra time series.
            """
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

        def construct_optimizer(random_variable, clip):
            parameters = {key: entry.evaluate() for key, entry in random_variable.parameters.items()}
            optim = optim_lib[random_variable.distribution](parameters)
            if clip != 0:
                optim = optax.chain(optax.clip(clip), optim)
            return optim

        # run multiple epochs
        for length, steps, optimizer, clip, make_step_epoch in zip(self.config.inference_optax.length_strategy, self.config.inference_optax.steps_strategy, self.config.inference_optax.optim_strategy, self.config.inference_optax.clip_strategy, make_step):

            batch_size = self.config.inference_optax.batch_size

            # create and initialize optimizer, chain with gradient clipping tool if clip parameter is not zero
            optim = construct_optimizer(optimizer, clip)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

            # determine which part of the observation data should be considered
            # (necessary so we don't need to JIT-compile for every epoch)
            length_eval = int(length_size * length)
            mask = jnp.concatenate([jnp.ones(ys.shape)[:batch_size,:length_eval], jnp.zeros(ys.shape)[:batch_size, length_eval :]], axis=1)
            t_thresh = ts[length_eval]

            # for data sets with more than one time series (require dataloader)
            if self.n_datasets > 1:
                # repeat procedure for every step
                for step, (yi,) in zip(
                    range(steps), dataloader((ys,), batch_size, key=loader_key)
                ):
                    # execute training step
                    loss, model, opt_state = make_step_epoch(ts, yi, x_in, mask, t_thresh, model, optim, opt_state, loss_func)

                    # create data for indepth modes
                    if self.config.inference_optax.indepth == "full":
                        model_list.append(model)
                    if self.config.inference_optax.indepth != "off":
                        lossev_single_model.append(loss)

                    # update progress bar
                    pbar.update(1)

                    # abort if time limit is exceeded or loss is infinite
                    current_time = time.time()
                    if not jnp.isfinite(loss).all() or current_time - start_time > self.config.inference_optax.time_limit:
                        if self.config.inference_optax.indepth == "partial":
                            return model, False, (current_time - start_time > self.config.inference_optax.time_limit), lossev_single_model
                        elif self.config.inference_optax.indepth == "full":
                            return model_list, False, (current_time - start_time > self.config.inference_optax.time_limit), lossev_single_model
                        else:
                            return None, False, (current_time - start_time > self.config.inference_optax.time_limit), None

            # for data sets with only one time series (do not require dataloader)
            else:
                # repeat procedure for every step
                for step, (yi,) in zip(
                    range(steps), [[ys]] * steps
                ):
                    # execute training step
                    loss, model, opt_state = make_step_epoch(ts, yi, x_in, mask, t_thresh, model, optim, opt_state, loss_func)

                    # create data for indepth modes
                    if self.config.inference_optax.indepth == "full":
                        model_list.append(model)
                    if self.config.inference_optax.indepth != "off":
                        lossev_single_model.append(loss)

                    # update progress bar
                    pbar.update(1)

                    # abort if time limit is exceeded or loss is infinite
                    current_time = time.time()
                    if not jnp.isfinite(loss).all() or current_time - start_time > self.config.inference_optax.time_limit:
                        if self.config.inference_optax.indepth == "partial":
                            return model, False, (current_time - start_time > self.config.inference_optax.time_limit), lossev_single_model
                        elif self.config.inference_optax.indepth == "full":
                            return model_list, False, (current_time - start_time > self.config.inference_optax.time_limit), lossev_single_model
                        else:
                            return None, False, (current_time - start_time > self.config.inference_optax.time_limit), None

        if self.config.inference_optax.indepth == "full":
            return model_list, True, False, lossev_single_model
        elif self.config.inference_optax.indepth == "partial":
            return model, True, False, lossev_single_model
        else:
            return model, True, False, None
    
    def optimize_model2(self, model, pbar, make_step):
        raise NotImplementedError()
        start_time = time.time()
        # transform observations to suitable format
        ts, ys, data_vars = self.transform_observations(self.simulation.observations)
        if self.n_datasets > 1:
            ys = ys[:self.n_train_sets]
        else:
            ys = jnp.expand_dims(ys,0)
        length_size = len(ts)

        # optimize model
        loader_key = jr.PRNGKey(np.random.randint(0,10000,()))

        last_model = model
        lossev_single_model = []

        def loss_func(y_obs, y_pred):
            return self.simulation.model.loss(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
        def dataloader(batch_size, *, key):
            indices = jnp.arange(self.n_train_sets)
            while True:
                perm = jr.permutation(key, indices)
                (key,) = jr.split(key, 1)
                start = 0
                end = batch_size
                while end < self.n_train_sets:
                    batch_perm = perm[start:end]
                    yield batch_perm
                    start = end
                    end = start + batch_size
        
        for length in self.config.inference_optax.length_strategy:

            clip = self.config.inference_optax.clip_strategy
            lr = self.config.inference_optax.lr_strategy
            steps = self.config.inference_optax.steps_strategy
            batch_size = self.config.inference_optax.batch_size

            if clip != 0:
                optim = optax.chain(optax.clip(clip), optax.adabelief(lr))
            else:
                optim = optax.adabelief(lr)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            length_eval = int(length_size * length)
            mask = jnp.concatenate([jnp.ones(ys.shape)[:batch_size,:length_eval], jnp.zeros(ys.shape)[:batch_size, length_eval :]], axis=1)
            evaluator = self.simulation.dispatch()

            if self.n_datasets > 1:
                for step, batch in zip(
                    range(steps), dataloader(self.config.inference_optax.batch_size, key=loader_key)
                ):
                    last_model = model
                    loss, model, opt_state = make_step(ys, batch, mask, model, evaluator, data_vars, opt_state, loss_func)
                    lossev_single_model.append(loss)
                    pbar.update(1)
                    current_time = time.time()
                    if not jnp.isfinite(loss).all() or current_time - start_time > 1200:
                        return last_model, False, lossev_single_model

            else:
                for step in range(steps):
                    last_model = model
                    loss, model, opt_state = make_step(ys, jnp.array([0]), mask, model, evaluator, data_vars, opt_state, loss_func)
                    lossev_single_model.append(loss)
                    pbar.update(1)
                    current_time = time.time()
                    if not jnp.isfinite(loss).all() or current_time - start_time > 1200:
                        return last_model, False, lossev_single_model

        return model, True, lossev_single_model
    
    def optimize_multiple_runs(self, make_step, key):
        """
        Runs the training process multiple times, as defined by the
        inference_optax.multiple_runs_target and inference_optax.multiple_runs_limit
        configuration options.

        Parameters
        ----------
        make_step : equinox._compile_utils.Compiled
            Compiled version of the make_step function.
        key : jax.ArrayImpl
            A key fixing all stochastic processes during training.
        
        Returns
        -------
        models : list[Any] or list[list[Any]]
            Depending on the value of the configuration option inference_optax.indepth, this
            contains either a list of the states of the models form each successful training
            instance after the last step of training or a nested list containing model states 
            after every step of training.
        failed_models : list[Any] or list[list[Any]]
            Depending on the value of the configuration option inference_optax.indepth, this
            contains either a list of the states of the models form each failed training
            instance after the last step of training or a nested list containing model states 
            after every step of training.
        success : list[bool]
            Contains one boolean entry for every attempt at training. If the corresponding
            training process was successful, this boolean is True, otherwise it is False.
        timeouts : int
            A number telling the user how many attempts of training timed out after the amount
            of time specified by the inference_optax.timeout configuration option. Only
            returned if the inference_optax.indepth configuration options is set to "partial"
            or "full".
        lossev : jax.ArrayImpl
            Depending on the value of the configuration option inference_optax.indepth, this
            returns either a None value or a nested array of loss values assumed over the 
            course of the multiple training processes.
        """
        cfg = self.config.inference_optax

        tried_runs = successful_runs = 0
        key = key

        models = []
        success = []
        if cfg.indepth:
            failed_models = []
            lossev = []
            timeouts = 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmWarning)

            pbar = tqdm(total = self.multiple_runs_target * jnp.sum(jnp.array(cfg.steps_strategy)).item(), desc=f"{successful_runs} of {self.multiple_runs_target} runs completed")

            # procedure is repeated until either the desired number of trained model or the limit of training runs is reached
            while tried_runs < cfg.multiple_runs_limit and successful_runs < self.multiple_runs_target:

                runstr = "run" if (tried_runs-successful_runs)==1 else "runs"
                pbar.set_postfix_str(f"{tried_runs - successful_runs} unsuccessful {runstr} so far")
                tried_runs += 1

                # constructs model with random initialization
                optimizable_model = self.construct_model(key)
                # trains model
                optimized_model, success_run, timeout, lossev_single_run = self.optimize_model(optimizable_model, pbar, make_step, key)

                # appends trained model either to the list containing successfully or unsuccessfully trained models and creates additional information
                if success_run:
                    models.append(optimized_model)
                    successful_runs += 1
                    pbar.set_description(f"{successful_runs} of {self.multiple_runs_target} runs completed")
                    success.append(True)
                    if cfg.indepth != "off":
                        lossev.append(lossev_single_run)
                        timeouts += timeout
                else:
                    success.append(False)
                    if cfg.indepth != "off":
                        failed_models.append(optimized_model)
                        lossev_single_run = lossev_single_run + [jnp.nan] * (jnp.sum(jnp.array(cfg.steps_strategy)).item() - len(lossev_single_run))
                        lossev.append(lossev_single_run)
                        timeouts += timeout
                    pbar.n = successful_runs * jnp.sum(jnp.array(cfg.steps_strategy)).item()
                    pbar.last_print_n = successful_runs * jnp.sum(jnp.array(cfg.steps_strategy)).item()

                # increments key for different results during next instance
                key = key+1

        if successful_runs < self.multiple_runs_target:
            warnings.warn(
                "Target number of successful runs was not reached before surpassing the " \
                f"allowed total number of runs. Only {successful_runs} optimized models were returned."
            )

        if cfg.indepth != "off":
            return models, failed_models, success, timeouts, jnp.array(lossev)
        else:
            return models, None, success, None, None
    
    def optimize_multiple_runs2(self, make_step):
        raise NotImplementedError()
        cfg = self.config.inference_optax

        tried_runs = successful_runs = 0

        models = []
        failed_models = []
        success = []
        lossev = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmWarning)

            pbar = tqdm(total = self.multiple_runs_target * jnp.sum(jnp.array(cfg.steps_strategy)).item() * len(cfg.length_strategy), desc=f"{successful_runs} of {self.multiple_runs_target} runs completed")

            while tried_runs < cfg.multiple_runs_limit and successful_runs < self.multiple_runs_target:

                runstr = "run" if (tried_runs-successful_runs)==1 else "runs"
                pbar.set_postfix_str(f"{tried_runs - successful_runs} unsuccessful {runstr} so far")
                tried_runs += 1
                
                # try:

                optimizable_model = self.construct_model()
                optimized_model, success_run, lossev_single_run = self.optimize_model2(optimizable_model, pbar, make_step)

                if success_run:
                    models.append(optimized_model)
                    successful_runs += 1
                    pbar.set_description(f"{successful_runs} of {self.multiple_runs_target} runs completed")
                    success.append(True)
                    lossev.append(lossev_single_run)
                else:
                    failed_models.append(optimized_model)
                    success.append(False)
                    lossev_single_run = lossev_single_run + [jnp.nan] * (cfg.steps_strategy * len(cfg.length_strategy) - len(lossev_single_run))
                    lossev.append(lossev_single_run)
                    pbar.n = successful_runs * cfg.steps_strategy * len(cfg.length_strategy)
                    pbar.last_print_n = successful_runs * cfg.steps_strategy * len(cfg.length_strategy)

                # except self.StopOptimizing:

                #     success.append(False)
                #     lossev_single_run = lossev_single_run + [jnp.nan] * (cfg.steps_strategy * len(cfg.length_strategy) - len(lossev_single_run))
                #     lossev.append(lossev_single_run)
                #     pbar.n = successful_runs * cfg.steps_strategy * len(cfg.length_strategy)
                #     pbar.last_print_n = successful_runs * cfg.steps_strategy * len(cfg.length_strategy)

                # except EquinoxRuntimeError:

                #     success.append(False)
                #     lossev_single_run = lossev_single_run + [jnp.nan] * (cfg.steps_strategy * len(cfg.length_strategy) - len(lossev_single_run))
                #     lossev.append(lossev_single_run)
                #     pbar.n = successful_runs * cfg.steps_strategy * len(cfg.length_strategy)
                #     pbar.last_print_n = successful_runs * cfg.steps_strategy * len(cfg.length_strategy)

        if successful_runs < self.multiple_runs_target:
            warnings.warn(
                "Target number of successful runs was not reached before surpassing the " \
                f"allowed total number of runs. Only {successful_runs} optimized models were returned."
            )

        return models, failed_models, success, jnp.array(lossev)
    
    def global_loss(self, model):
        """
        Returns the loss of the passed model with respect to every time series from the
        validation data.

        Parameters
        ----------
        model : Any
            Model object containing all model parameters somewhere within a pytree
            structure and returning derivatives depending on the model states when
            called with the model states as input.

        Returns
        -------
        loss : jax.ArrayImpl
            A jax.numpy array containing the loss of the passed model across all of the
            validation data as its only entry.       
        """
        ts, ys, data_vars = self.transform_observations(self.simulation.observations)
        if self.n_datasets > 1:
            ys = ys[self.n_train_sets:]
        else:
            ys = jnp.expand_dims(ys,0)

        if "x_in" in self.simulation.model_parameters.keys() and [x for x in self.simulation.model_parameters["x_in"].data_vars] != []:
            x_in_temp = self.transform_x_in(self.simulation.model_parameters["x_in"])
            x_in = (x_in_temp[0], x_in_temp[1][0])
        else:
            x_in = None

        def loss_func(y_obs, y_pred):
            return self.simulation.model.loss(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
        @eqx.filter_jit
        def loss(model, ti, yi, loss_func):
            y_pred = jnp.array(jax.vmap(self.simulation.evaluator._solver.standalone_solver, in_axes=(None, None, 0, None, None))(model, ti, yi[:, 0], x_in, ti[-1]))
            y_pred = jnp.stack(y_pred, axis = (len(y_pred.shape)-1))

            losses = loss_func(yi, y_pred)
            return jnp.mean(losses)
        
        global_loss = loss(model, ts, ys, loss_func)

        return global_loss
    
    def global_loss2(self, model):
        raise NotImplementedError()
        ts, ys, data_vars = self.transform_observations(self.simulation.observations)
        if self.n_datasets > 1:
            ys = ys[self.n_train_sets:]
        else:
            ys = jnp.expand_dims(ys,0)

        def loss_func(y_obs, y_pred):
            return self.simulation.model.loss(jnp.where(jnp.isnan(y_obs), y_pred, y_obs), y_pred)
            
        @eqx.filter_jit
        def loss(model, ti, yi, evaluator, loss_func):
            evaluator.model = model
            evaluator()
            y_pred = jnp.array([evaluator.Y[data_var] for data_var in data_vars])
            y_pred = jnp.stack(y_pred, axis = (len(y_pred.shape)-1))

            if self.n_datasets > 1:
                y_pred = y_pred[self.n_train_sets:]

            losses = loss_func(yi, y_pred)
            return jnp.mean(losses)
        
        evaluator = self.simulation.dispatch()
        
        return loss(model, ts, ys, evaluator, loss_func)
    
    def sort_models_by_global_loss(self, models):
        """
        Sorts models by their loss across all of the validation data.

        Parameters
        ----------
        models : list[Any]
            List of model objects containing all model parameters somewhere within a pytree
            structure and returning derivatives depending on the model states when called
            with the model states as input.

        Returns
        -------
        sorted_indices : list[int]
            List of indices of the passed list of models sorted by losses in ascending order,
            that is, the first index from this list represents the position of the model with
            the lowest loss, and so on.
        """
        losses = [self.global_loss(model) for model in models]

        sorted_losses = []
        sorted_indices = []

        for x, y in sorted(zip(losses, [i for i in range(len(losses))])):
            sorted_losses.append(x)
            sorted_indices.append(y)

        return sorted_indices
    
    def sort_models_by_global_loss2(self, models):
        raise NotImplementedError()
        losses = [self.global_loss2(model) for model in models]

        sorted_losses = []
        sorted_indices = []

        for x, y in sorted(zip(losses, [i for i in range(len(losses))])):
            sorted_losses.append(x)
            sorted_indices.append(y)

        return sorted_indices

    def create_idata(self):
        """
        Creates inference data for trained models. Inference data contains information
        about the posterior model parameter values, the observation data, predictions
        made by the trained model, and the errors of these predictions.

        Returns
        -------
        idata : xarray.core.dataset.Dataset
            Inference data for successfully trained models.
        idata_f : xarray.core.dataset.Dataset
            Inference data for models from failed instances of training. Only returned if 
            the inference_optax.indepth is set to "partial" (returns data for the last
            state of models before failure) or "full" (returns data for all intermediary
            states of training); returns None otherwise. Inference data for failed runs is 
            reduced, that is, only posterior parameter values are included.
        """
        list = [key for key in self.simulation.config.model_parameters.free.keys()]
        ts, ys, data_vars = self.transform_observations(self.simulation.observations)
        if self.n_datasets == 1:
            ys = jnp.expand_dims(ys, 0)
        batch_ids = jnp.arange(self.n_datasets)
        chain_ids = jnp.arange(1)

        if self.config.inference_optax.indepth == "full" and len(self.optimized_models) > 0:

            dict = {}
            for entry in list:
                dict[entry] = np.array([[[getattr(model, entry) for model in models] + [np.nan]*(sum(self.config.inference_optax.steps_strategy) + 1 - len(models)) for models in self.optimized_models]])

            reference_model = self.simulation.model
            mlp_size = (reference_model.mlp.in_size, reference_model.mlp.out_size, reference_model.mlp.width_size, reference_model.mlp.depth)

            len_weights = mlp_size[0]*mlp_size[2] + (mlp_size[3] - 1)*mlp_size[2]**2 + mlp_size[2]*mlp_size[1]
            len_bias = mlp_size[3]*mlp_size[2] + mlp_size[1]

            dict["weights"] = np.array([[[transformWeights(getFuncWeights(model))[4] for model in self.optimized_models[i]] + [[np.nan] * len_weights] * (sum(self.config.inference_optax.steps_strategy) + 1 - len(self.optimized_models[i])) for i in np.arange(len(self.optimized_models))]])
            dict["bias"] = np.array([[[transformBias(getFuncBias(model))[3] for model in self.optimized_models[i]] + [[np.nan] * len_bias] * (sum(self.config.inference_optax.steps_strategy) + 1 - len(self.optimized_models[i])) for i in np.arange(len(self.optimized_models))]])

            dims = {entry: ["draw","step"] for entry in list}
            dims["weights"] = ["chain","draw","step","n_weight"]
            dims["bias"] = ["chain","draw","step","n_bias"]

            idata = az.convert_to_inference_data(
                dict,
                dims = dims,
                coords = {"n_weight": np.arange(len(dict["weights"][0,0,0])), "n_bias": np.arange(len(dict["bias"][0,0,0])), "step": np.arange(sum(self.config.inference_optax.steps_strategy) + 1)}
            )

            post_pred = {}
            losses = {}
            data_vars = self.simulation.observations.data_vars
            evaluator = self.simulation.dispatch()
            for x in data_vars:
                post_pred[x] = []
                losses[x] = []

            for model in self.optimized_models:
                sol = jnp.array([evaluator._solver.standalone_solver(model[-1], ts, y0[0], (), ts[-1]) for y0 in ys])
                for i, x in enumerate(data_vars):
                    post_pred[x].append(sol[:,i])
                    losses[x].append(self.simulation.model.loss(self.simulation.observations[x].values, sol[:,i]))

            for x in data_vars:
                post_pred[x] = jnp.array(post_pred[x])
                post_pred[x] = jnp.expand_dims(post_pred[x], 0)
                losses[x] = jnp.array(losses[x])
                losses[x] = jnp.expand_dims(losses[x], 0)

            post_pred_xr = []
            losses_xr = []

            model_ids = jnp.arange(len(self.optimized_models))

            for x in data_vars:
                post_pred_xr.append(xr.DataArray(post_pred[x], coords={"chain": chain_ids, "draw": model_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))
                losses_xr.append(xr.DataArray(losses[x], coords={"chain": chain_ids, "draw": model_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))

            post_pred_xr = xr.merge([x for x in post_pred_xr])
            losses_xr = xr.merge([x for x in losses_xr])

            idata.add_groups({"observed_data": self.simulation.observations, "posterior_model_fits": post_pred_xr, "losses": losses_xr})
            idata.add_groups({"posterior_predictive": idata.posterior_model_fits, "log_likelihood": idata.losses})

        elif len(self.optimized_models) > 0:

            dict = {list[j]: np.array([getattr(self.optimized_models[i], list[j]) for i in np.arange(len(self.optimized_models))]) for j in np.arange(len(list))}
            dict["weights"] = np.array([[transformWeights(getFuncWeights(model))[4] for model in self.optimized_models]])
            dict["bias"] = np.array([[transformBias(getFuncBias(model))[3] for model in self.optimized_models]])

            idata = az.convert_to_inference_data(
                dict,
                dims = {"weights": ["chain","draw","n_weight"], "bias": ["chain","draw","n_bias"]},
                coords = {"n_weight": np.arange(len(dict["weights"][0,0])), "n_bias": np.arange(len(dict["bias"][0,0]))}
            )

            post_pred = {}
            losses = {}
            data_vars = self.simulation.observations.data_vars
            evaluator = self.simulation.dispatch()
            for x in data_vars:
                post_pred[x] = []
                losses[x] = []

            for model in self.optimized_models:
                sol = jnp.array([evaluator._solver.standalone_solver(model, ts, y0[0], (), ts[-1]) for y0 in ys])
                for i, x in enumerate(data_vars):
                    post_pred[x].append(sol[:,i])
                    losses[x].append(self.simulation.model.loss(self.simulation.observations[x].values, sol[:,i]))

            for x in data_vars:
                post_pred[x] = jnp.array(post_pred[x])
                post_pred[x] = jnp.expand_dims(post_pred[x], 0)
                losses[x] = jnp.array(losses[x])
                losses[x] = jnp.expand_dims(losses[x], 0)

            post_pred_xr = []
            losses_xr = []

            model_ids = jnp.arange(len(self.optimized_models))

            for x in data_vars:
                post_pred_xr.append(xr.DataArray(post_pred[x], coords={"chain": chain_ids, "draw": model_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))
                losses_xr.append(xr.DataArray(losses[x], coords={"chain": chain_ids, "draw": model_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))

            post_pred_xr = xr.merge([x for x in post_pred_xr])
            losses_xr = xr.merge([x for x in losses_xr])

            idata.add_groups({"observed_data": self.simulation.observations, "posterior_model_fits": post_pred_xr, "losses": losses_xr})
            idata.add_groups({"posterior_predictive": idata.posterior_model_fits, "log_likelihood": idata.losses})

        else:

            idata = None

        if self.config.inference_optax.indepth == "partial" and len(self.failed_models) > 0:

            dict_f = {list[j]: np.array([getattr(self.failed_models[i], list[j]) for i in np.arange(len(self.failed_models))]) for j in np.arange(len(list))}
            dict_f["weights"] = np.array([[transformWeights(getFuncWeights(model))[4] for model in self.failed_models]])
            dict_f["bias"] = np.array([[transformBias(getFuncBias(model))[3] for model in self.failed_models]])

            idata_f = az.convert_to_inference_data(
                dict_f,
                dims = {"weights": ["chain","draw","n_weight"], "bias": ["chain","draw","n_bias"]},
                coords = {"n_weight": np.arange(len(dict_f["weights"][0,0])), "n_bias": np.arange(len(dict_f["bias"][0,0]))}
            )

            # post_pred_f = {}
            # losses_f = {}
            # data_vars = self.simulation.observations.data_vars
            # evaluator = self.simulation.dispatch()
            # for x in data_vars:
            #     post_pred_f[x] = []
            #     losses_f[x] = []
            
            # for model in self.failed_models:
            #     sol = jnp.array([evaluator._solver.standalone_solver(model, ts, y0[0], (), ts[-1]) for y0 in ys])
            #     for i, x in enumerate(data_vars):
            #         post_pred_f[x].append(sol[:,i])
            #         losses_f[x].append(self.simulation.model.loss(self.simulation.observations[x].values, sol[:,i]))

            # for x in data_vars:
            #     post_pred_f[x] = jnp.array(post_pred_f[x])
            #     post_pred_f[x] = jnp.expand_dims(post_pred_f[x], 0)
            #     losses_f[x] = jnp.array(losses_f[x])
            #     losses_f[x] = jnp.expand_dims(losses_f[x], 0)

            # post_pred_f_xr = []
            # losses_f_xr = []

            # model_f_ids = jnp.arange(len(self.failed_models))

            # for x in data_vars:
            #     post_pred_f_xr.append(xr.DataArray(post_pred_f[x], coords={"chain": chain_ids, "draw": model_f_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))
            #     losses_f_xr.append(xr.DataArray(losses_f[x], coords={"chain": chain_ids, "draw": model_f_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))

            # post_pred_f_xr = xr.merge([x for x in post_pred_f_xr])
            # losses_f_xr = xr.merge([x for x in losses_f_xr])

            # idata_f.add_groups({"observed_data": self.simulation.observations, "posterior_model_fits": post_pred_f_xr, "losses": losses_f_xr})
            # idata_f.add_groups({"posterior_predictive": idata_f.posterior_model_fits, "log_likelihood": idata_f.losses})

        elif self.config.inference_optax.indepth == "full" and len(self.failed_models) > 0:

            dict_f = {}
            for entry in list:
                dict_f[entry] = np.array([[[getattr(model, entry) for model in models] + [np.nan]*(sum(self.config.inference_optax.steps_strategy) + 1 - len(models)) for models in self.failed_models]])

            reference_model = self.simulation.model
            mlp_size = (reference_model.mlp.in_size, reference_model.mlp.out_size, reference_model.mlp.width_size, reference_model.mlp.depth)

            len_weights = mlp_size[0]*mlp_size[2] + (mlp_size[3] - 1)*mlp_size[2]**2 + mlp_size[2]*mlp_size[1]
            len_bias = mlp_size[3]*mlp_size[2] + mlp_size[1]

            dict_f["weights"] = np.array([[[transformWeights(getFuncWeights(model))[4] for model in self.failed_models[i]] + [[np.nan] * len_weights] * (sum(self.config.inference_optax.steps_strategy) + 1 - len(self.failed_models[i])) for i in np.arange(len(self.failed_models))]])
            dict_f["bias"] = np.array([[[transformBias(getFuncBias(model))[3] for model in self.failed_models[i]] + [[np.nan] * len_bias] * (sum(self.config.inference_optax.steps_strategy) + 1 - len(self.failed_models[i])) for i in np.arange(len(self.failed_models))]])

            dims_f = {entry: ["draw","step"] for entry in list}
            dims_f["weights"] = ["chain","draw","step","n_weight"]
            dims_f["bias"] = ["chain","draw","step","n_bias"]

            idata_f = az.convert_to_inference_data(
                dict_f,
                dims = dims_f,
                coords = {"n_weight": np.arange(len(dict_f["weights"][0,0,0])), "n_bias": np.arange(len(dict_f["bias"][0,0,0])), "step": np.arange(sum(self.config.inference_optax.steps_strategy) + 1)}
            )

            # post_pred_f = {}
            # losses_f = {}
            # data_vars = self.simulation.observations.data_vars
            # evaluator = self.simulation.dispatch()
            # for x in data_vars:
            #     post_pred_f[x] = []
            #     losses_f[x] = []
            
            # for models in self.failed_models:
            #     sols = jnp.array([[evaluator._solver.standalone_solver(model, ts, y0[0], (), ts[-1]) for y0 in ys] for model in models]+[[[[np.nan]*len(ts)]]*len(ys)] * (sum(self.config.inference_optax.steps_strategy) + 1 - len(models)))
            #     for i, x in enumerate(data_vars):
            #         post_pred_f[x].append([sol[:,i] for sol in sols])
            #         losses_f[x].append([self.simulation.model.loss(self.simulation.observations[x].values, sol[:,i]) for sol in sols])

            # for x in data_vars:
            #     post_pred_f[x] = jnp.array(post_pred_f[x])
            #     post_pred_f[x] = jnp.expand_dims(post_pred_f[x], 0)
            #     losses_f[x] = jnp.array(losses_f[x])
            #     losses_f[x] = jnp.expand_dims(losses_f[x], 0)

            # post_pred_f_xr = []
            # losses_f_xr = []

            # model_f_ids = jnp.arange(len(self.failed_models))

            # for x in data_vars:
            #     post_pred_f_xr.append(xr.DataArray(post_pred_f[x], coords={"chain": chain_ids, "draw": model_f_ids, "step": np.arange(sum(self.config.inference_optax.steps_strategy) + 1), "data_batch": batch_ids, "time": ts}).to_dataset(name=x))
            #     losses_f_xr.append(xr.DataArray(losses_f[x], coords={"chain": chain_ids, "draw": model_f_ids, "step": np.arange(sum(self.config.inference_optax.steps_strategy) + 1), "data_batch": batch_ids, "time": ts}).to_dataset(name=x))

            # post_pred_f_xr = xr.merge([x for x in post_pred_f_xr])
            # losses_f_xr = xr.merge([x for x in losses_f_xr])

            # idata_f.add_groups({"observed_data": self.simulation.observations, "posterior_model_fits": post_pred_f_xr, "losses": losses_f_xr})
            # idata_f.add_groups({"posterior_predictive": idata_f.posterior_model_fits, "log_likelihood": idata_f.losses})

        else:

            idata_f = None

        return idata, idata_f
    
    def create_idata2(self):
        raise NotImplementedError()
        list = [key for key in self.simulation.config.model_parameters.free.keys()]
        ts = self.simulation.observations.time.values
        batch_ids = jnp.arange(self.n_datasets)
        chain_ids = jnp.arange(1)

        if len(self.optimized_models) > 0:

            dict = {list[j]: np.array([getattr(self.optimized_models[i], list[j]) for i in np.arange(len(self.optimized_models))]) for j in np.arange(len(list))}
            dict["weights"] = np.array([[transformWeights(getFuncWeights(model))[4] for model in self.optimized_models]])
            dict["bias"] = np.array([[transformBias(getFuncBias(model))[3] for model in self.optimized_models]])

            idata = az.convert_to_inference_data(
                dict,
                dims = {"weights": ["chain","draw","n_weight"], "bias": ["chain","draw","n_bias"]},
                coords = {"n_weight": np.arange(len(dict["weights"][0,0])), "n_bias": np.arange(len(dict["bias"][0,0]))}
            )

            post_pred = {}
            losses = {}
            data_vars = self.simulation.observations.data_vars
            evaluator = self.simulation.dispatch()
            for x in data_vars:
                post_pred[x] = []
                losses[x] = []

            for model in self.optimized_models:
                evaluator.model = model
                evaluator()
                
                for x in data_vars:
                    post_pred[x].append(evaluator.Y[x])
                    losses[x].append(self.simulation.model.loss(self.simulation.observations[x].values, evaluator.Y[x]))

            for x in data_vars:
                post_pred[x] = jnp.array(post_pred[x])
                post_pred[x] = jnp.expand_dims(post_pred[x], 0)
                losses[x] = jnp.array(losses[x])
                losses[x] = jnp.expand_dims(losses[x], 0)

            post_pred_xr = []
            losses_xr = []

            model_ids = jnp.arange(len(self.optimized_models))

            for x in data_vars:
                if self.n_datasets == 1:
                    post_pred[x] = jnp.expand_dims(post_pred[x], 2)
                    losses[x] = jnp.expand_dims(losses[x], 2)
                post_pred_xr.append(xr.DataArray(post_pred[x], coords={"chain": chain_ids, "draw": model_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))
                losses_xr.append(xr.DataArray(losses[x], coords={"chain": chain_ids, "draw": model_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))

            post_pred_xr = xr.merge([x for x in post_pred_xr])
            losses_xr = xr.merge([x for x in losses_xr])

            idata.add_groups({"observed_data": self.simulation.observations, "posterior_model_fits": post_pred_xr, "losses": losses_xr})
            idata.add_groups({"posterior_predictive": idata.posterior_model_fits, "log_likelihood": idata.losses})

        else:

            idata = None

        if len(self.failed_models) > 0:

            dict_f = {list[j]: np.array([getattr(self.failed_models[i], list[j]) for i in np.arange(len(self.failed_models))]) for j in np.arange(len(list))}
            dict_f["weights"] = np.array([[transformWeights(getFuncWeights(model))[4] for model in self.failed_models]])
            dict_f["bias"] = np.array([[transformBias(getFuncBias(model))[3] for model in self.failed_models]])

            idata_f = az.convert_to_inference_data(
                dict_f,
                dims = {"weights": ["chain","draw","n_weight"], "bias": ["chain","draw","n_bias"]},
                coords = {"n_weight": np.arange(len(dict_f["weights"][0,0])), "n_bias": np.arange(len(dict_f["bias"][0,0]))}
            )

            post_pred_f = {}
            losses_f = {}
            data_vars = self.simulation.observations.data_vars
            evaluator = self.simulation.dispatch()
            for x in data_vars:
                post_pred_f[x] = []
                losses_f[x] = []
            
            for model in self.failed_models:
                evaluator.model = model
                evaluator()
                
                for x in data_vars:
                    post_pred_f[x].append(evaluator.Y[x])
                    losses_f[x].append(self.simulation.model.loss(self.simulation.observations[x].values, evaluator.Y[x]))

            for x in data_vars:
                post_pred_f[x] = jnp.array(post_pred_f[x])
                post_pred_f[x] = jnp.expand_dims(post_pred_f[x], 0)
                losses_f[x] = jnp.array(losses_f[x])
                losses_f[x] = jnp.expand_dims(losses_f[x], 0)

            post_pred_f_xr = []
            losses_f_xr = []

            model_f_ids = jnp.arange(len(self.failed_models))

            for x in data_vars:
                if self.n_datasets == 1:
                    post_pred_f[x] = jnp.expand_dims(post_pred_f[x], 2)
                    losses_f[x] = jnp.expand_dims(losses_f[x], 2)
                post_pred_f_xr.append(xr.DataArray(post_pred_f[x], coords={"chain": chain_ids, "draw": model_f_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))
                losses_f_xr.append(xr.DataArray(losses_f[x], coords={"chain": chain_ids, "draw": model_f_ids, "data_batch": batch_ids, "time": ts}).to_dataset(name=x))

            post_pred_f_xr = xr.merge([x for x in post_pred_f_xr])
            losses_f_xr = xr.merge([x for x in losses_f_xr])

            idata_f.add_groups({"observed_data": self.simulation.observations, "posterior_model_fits": post_pred_f_xr, "losses": losses_f_xr})
            idata_f.add_groups({"posterior_predictive": idata_f.posterior_model_fits, "log_likelihood": idata_f.losses})

        else:

            idata_f = None

        return idata, idata_f
    
    def store_results(self, output=None, output_f=None):
        """
        Stores inference data at the specified location.

        Parameters
        ----------
        output : str
            Location to which the inference data of successful instances of training should
            be stored. If set to None, it is saved to the location specified by the
            output_path attribute of the SimulationBase object at self.simulation.
        output_f : str
            Location to which the inference data of failed instances of training should be
            stored. If set to None, it is saved to the location specified by the output_path
            attribute of the SimulationBase object at self.simulation.  
        """
        if self.idata != None:
            if output is not None:
                self.idata.to_netcdf(output)
            else:
                self.idata.to_netcdf(f"{self.simulation.output_path}/optax_idata.nc")
        if self.idata_f != None:
            if output_f is not None:
                self.idata_f.to_netcdf(output_f)
            else:
                self.idata_f.to_netcdf(f"{self.simulation.output_path}/optax_idata_f.nc")

    def load_results(self, file="optax_idata.nc"):
        """
        Loads inference data from the specified location and saves it at self.idata.

        Parameters
        ----------
        file : str
            File name of the NetCDF file containing the inference data. The file needs to
            be located within the directory specified by the output_path attribute of the 
            SimulationBase object at self.simulation.
        """
        idata = az.from_netcdf(f"{self.simulation.output_path}/{file}")

        self.idata = idata

    def store_loss_evolution(self, output=None):
        """
        Stores loss evolution data at the specified location.

        Parameters
        ----------
        output : str
            Location to which the loss evolution data should be stored. If set to None, it 
            is saved to the location specified by the output_path attribute of the 
            SimulationBase object at self.simulation.
        """
        if output is not None:
            self.lossev.to_netcdf(output)
        else:
            self.lossev.to_netcdf(f"{self.simulation.output_path}/loss_evolution.nc")
    
    def load_loss_evolution(self, file="loss_evolution.nc", cluster: Optional[int] = None):
        """
        Loads loss evolution data from the specified location and saves it at self.lossev.

        Parameters
        ----------
        file : str
            File name of the NetCDF file containing the loss evolution data. The file needs 
            to be located within the directory specified by the output_path attribute of the 
            SimulationBase object at self.simulation.
        """
        lossev = xr.open_dataset(f"{self.simulation.output_path}/{file}")
        if cluster is not None:
            self.select_cluster(lossev, cluster)

        self.lossev = lossev
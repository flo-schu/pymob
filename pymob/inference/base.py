import numpy as np
import ast
from typing import (
    Dict,
    Tuple,
    Mapping,
    Callable,
    Iterable,
    Optional,
    Any,
    List,
    Protocol
)
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
import arviz as az
import itertools as it
import pandas as pd

from pymob.simulation import SimulationBase
from pymob.sim.parameters import Param, RandomVariable, Expression
from pymob.utils.config import lookup_from

class Errorfunction(Protocol):
    def __call__(
        self, 
        theta: Dict[str, List[float]], 
    ) -> Any:
        ...

class Distribution:
    """The distribution is a pre-initialized distibution with human friendly 
    interface to construct a distribution in an arbitrary backend.

    The necessary adjustments to make the distribution backend specific
    are done by passing more context to Distribution class to the
    _context variable
    """
    distribution_map: Dict[str,Tuple[Callable, Dict[str,str]]] = {}
    parameter_converter: Callable = staticmethod(lambda x: x)
    _context = {}
    _import_map = {"np": "numpy", "numpy":"numpy", "jnp":"jax.numpy", "jax.numpy": "jax.numpy"}

    def __init__(
        self, 
        name: str, 
        random_variable: RandomVariable, 
        dims: Tuple[str, ...],
        shape: Tuple[int, ...],
    ) -> None:
        self.name = name
        self._dist_str = random_variable.distribution
        self._parameter_expression = random_variable.parameters
        self._obs_transform: Optional[Expression] = random_variable.obs
        self.dims = dims
        self.shape = shape if len(shape) > 0 else ()

        dist, params, uargs = self.parse_distribution(random_variable)
        self.distribution: Callable = dist
        self.parameters: Dict[str, Expression] = params
        self.undefined_args: set = uargs

        self.create_obs_transform_func()

    def __str__(self) -> str:
        dist = self.dist_name
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        dimshape = tuple([f'{d}={s}' for d, s in zip(self.dims, self.shape)])
        return f"{dist}({params}, dims={dimshape}, obs={self._obs_transform})"
    
    def __repr__(self) -> str:
        return str(self)

    @property
    def dist_name(self) -> str:
        return self._dist_str
    
    def create_obs_transform_func(self):
        if self._obs_transform is None:
            self._obs_transform_module = None
            self.obs_transform_func = None
        
        else:
            expr = self._obs_transform
            
            imports = [a for a in expr.undefined_args if a in self._import_map]
            args = [a for a in expr.undefined_args if a not in self._import_map]

            # Create the function arguments
            args = [ast.arg(arg=arg_name, annotation=None) for arg_name in args]
            
            # Build a function definition from the expression
            func_def = ast.FunctionDef(
                name=f"obs_transform_{self.name}",
                args=ast.arguments(
                    posonlyargs=[], args=args, vararg=None, kwonlyargs=[], 
                    kw_defaults=[], defaults=[]
                ),
                body=[ast.Return(value=expr.expression.body)],
                decorator_list=[]
            )
        
            import_statements = [
                ast.Import(names=[ast.alias(name=self._import_map[i], asname=i)])
                for i in imports
            ]
            module = ast.Module(body=[*import_statements, func_def], type_ignores=[])
            module = ast.fix_missing_locations(module)

            # compile the function and retrieve object
            code = compile(module, filename="<ast>", mode="exec")
            func_env = {}
            exec(code, func_env)    
            func = func_env[f"obs_transform_{self.name}"]
            
            self._obs_transform_module = module
            self.obs_transform_func = func

    
    def construct(self, context: Iterable[Mapping], extra_kwargs: Dict = {}):
        _context = {arg: lookup_from(arg, context) for arg in self.undefined_args}
        _context.update(self._context)
        # evaluate the parameters given a context
        params = {
            key: self.parameter_converter(value.evaluate(context=_context)) 
            for key, value in self.parameters.items()
        }
        return self.distribution(**params, **extra_kwargs)

    def _get_distribution(self, distribution: str) -> Tuple[Callable, Dict[str, str]]:
        return self.distribution_map[distribution]

    def parse_distribution(self, random_variable: RandomVariable) -> Tuple[Any,Dict[str,Expression],set]:

        distribution_mapping = self._get_distribution(random_variable.distribution)

        if not isinstance(distribution_mapping, tuple):
            distribution = distribution_mapping
            distribution_mapping = (distribution, {})
        
        assert len(distribution_mapping) == 2, (
            "distribution and parameter mapping must be "
            "a tuple of length 2."
        )

        distribution, parameter_mapping = distribution_mapping
        mapped_params = {}
        underfined_args = set()
        for key, val in random_variable.parameters.items():
            mapped_key = parameter_mapping.get(key, key)
            underfined_args = underfined_args.union(val.undefined_args)
            mapped_params.update({mapped_key:val})

        return distribution, mapped_params, underfined_args


class InferenceBackend(ABC):
    _distribution = Distribution
    idata: az.InferenceData
    prior: Dict[str,Distribution]
    log_likelihood: Errorfunction
    gradient_log_likelihood: Errorfunction

    def __init__(
        self, 
        simulation: SimulationBase,
    ) -> None:
        
        self.simulation = simulation
        self.config = simulation.config

        self.indices = {v.name: list(v.values) for _, v in self.simulation.indices.items()}
        # parse model components
        self.prior = self.parse_model_priors(
            parameters=self.config.model_parameters.free,
            dim_shapes=self.simulation.parameter_shapes,
            indices=self.indices
        )

        self.evaluator = self.parse_deterministic_model()

        self.error_model = self.parse_error_model(
            error_models=self.config.error_model.all
        )
        

    @abstractmethod
    def parse_deterministic_model(self):
        pass

    @abstractmethod
    def parse_probabilistic_model(self):
        pass

    @property
    def extra_vars(self):
        return self.config.inference.extra_vars
    
    @property
    def n_predictions(self):
        return self.config.inference.n_predictions
    
    @property
    def EPS(self):
        return self.config.inference.eps

    @property
    def posterior_data_structure(self) -> Dict[str, List[str]]:
        data_structure = self.simulation.data_structure.copy()
        data_structure_loglik = {f"{dv}_obs": dims for dv, dims in data_structure.items()}
        parameter_dims = {k: list(v) for k, v in self.simulation.parameter_dims.items() if len(v) > 0}
        data_structure.update(data_structure_loglik)
        data_structure.update(parameter_dims)
        return data_structure
    
    @property
    def posterior_coordinates(self) -> Dict[str, List[str|int]]:
        if not hasattr(self, "chains"):
            chains = 1
        else:
            chains = self.chains

        if not hasattr(self, "draws"):
            draws = 1
        else:
            draws = self.draws

        posterior_coords = {k: list(v) for k, v in self.simulation.dimension_coords.items()}
        posterior_coords.update({
            "draw": list(range(draws)), 
            "chain": list(range(chains))
        })
        return posterior_coords

    @classmethod
    def parse_model_priors(
        cls, 
        parameters: Dict[str,Param], 
        dim_shapes: Dict[str,Tuple[int, ...]],
        indices: Dict[str, Any] = {}
    ):
        priors = {}
        hyper_ = []
        for key, par in parameters.items():
            if par.prior is None:
                raise AttributeError(
                    f"No prior was defined for parameter '{key}'. E.g.: "+
                    f"`sim.config.model_parameters.{key}.prior = 'lognorm(loc=1, scale=2)'`"
                )
            
            for k, v in par.prior.parameters.items():
                for ua in v.undefined_args:
                    if ua in indices:
                        continue

                    elif ua in cls._distribution._import_map:
                        continue

                    elif ua in priors:
                        continue

                    else:
                        raise KeyError(
                            f"Parameter '{key}' defines a prior '{par.prior.model_ser()}' that will try "+
                            f"to access the variable '{ua}' before it is defined "+
                            "in 'sim.indices', 'Distribution._import_map' or previously "+
                            "defined priors. Please double check the prior definition for errors, "+
                            f"specify the needed parameter, or check the parameter order. "+
                            "If needed, use "+
                            f"config.model_parameters.reorder([..., '{ua}', '{key}', ...]) "+
                            "to arrange the parameters in the correct order."
                        )

            dist = cls._distribution(
                name=key, 
                random_variable=par.prior,
                dims=par.dims,
                shape=dim_shapes[key]
            )
            priors.update({key: dist})
        return priors
    
    @classmethod
    def parse_error_model(
        cls,
        error_models: Dict[str,RandomVariable], 
    ):
        error_model = {}
        for data_var, error_distribution in error_models.items():
            error_dist = cls._distribution(
                name=data_var, 
                random_variable=error_distribution,
                dims=(),
                shape=(),
            )
               
            error_model.update({data_var: error_dist})
        return error_model

    @abstractmethod
    def create_log_likelihood(self) -> Tuple[Errorfunction,Errorfunction]:
        """This method creates a log likelihood function and potentially
        function to compute the gradients.
        """
        pass

    def plot_likelihood_landscape(
        self, 
        parameters: Tuple[str, str],
        log_likelihood_func: Callable,
        gradient_func: Optional[Callable] = None,
        bounds: Tuple[List[float],List[float]] = ([-10, 10], [-10,10]),
        n_grid_points: int = 100,
        n_vector_points: int = 20,
        ax: Optional[plt.Axes] = None,
    ):
        """Plots the likelihood for each coordinate pair of two model parameters
        Parameters are taken from the standardized scale and transformed 

        Parameters
        ----------

        parameters : Tuple[str,str]
            The parameters to be plotted against each other

        log_likelihood_func : Callable
            Must be a vectorized function that can dictionary with two keys that
            contain array values as input parameters and returns a 1-D array of 
            log-likelihood values.

        gradient_func : Callable
            If, in addition to the likelihood values, also gradients should
            be computed, a function to compute the gradients must be provided 

        bounds : Tuple[Tuple[float,float],Tuple[float,float]]
            Fallback bounds for the parameter grid to calculate the likelihood
            function. This is only used if bounds are not provided by the parameters

        n_grid_points : int
            The number of grid points per side, to calculate the likelihood
            function for. Scales n^2
        """
        par_x, par_y = parameters
        bounds_x, bounds_y = bounds
        px = self.config.model_parameters.all[par_x]
        py = self.config.model_parameters.all[par_y]

        bounds_x[0] = px.min if px.min is not None else bounds_x[0]
        bounds_x[1] = px.max if px.max is not None else bounds_x[1]
        bounds_y[0] = py.min if py.min is not None else bounds_y[0]
        bounds_y[1] = py.max if py.max is not None else bounds_y[1]

        x = np.linspace(*bounds_x, n_grid_points)
        y = np.linspace(*bounds_y, n_grid_points)

        grid = {p: v for p, v in zip(parameters, np.array(list(it.product(x, y))).T)}
        loglik = log_likelihood_func(grid)

        Z = loglik.reshape((n_grid_points, n_grid_points))
        X, Y = np.meshgrid(x, y)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        contours = ax.contourf(X, Y, Z)

        if gradient_func is not None:
            xv = np.linspace(*bounds_x, n_vector_points)
            yv = np.linspace(*bounds_y, n_vector_points)
            Xv, Yv = np.meshgrid(xv, yv)

            gridv = {p: v for p, v in zip(parameters, np.array(list(it.product(xv, yv))).T)}

            grads = gradient_func(gridv)
            U, V = grads[par_x], grads[par_y]
            ax.quiver(Xv, Yv, U, V)

        plt.colorbar(contours, label="log-likelihood")
        ax.set_xlabel("beta (normalized)")
        ax.set_ylabel("alpha (normalized)")

        return ax
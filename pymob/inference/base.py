import re
from typing import (
    Dict,
    Tuple,
    Literal,
    Union,
    Mapping,
    Callable,
    Iterable,
    Optional,
    Any
)
from abc import ABC, abstractmethod

import arviz as az

from pymob.simulation import SimulationBase
from pymob.sim.parameters import Param, RandomVariable, Expression
from pymob.sim.config import Modelparameters
from pymob.utils.config import lookup_from

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
        self.dims = dims
        self.shape = shape if len(shape) > 0 else ()

        dist, params, uargs = self.parse_distribution(random_variable)
        self.distribution: Callable = dist
        self.parameters: Dict[str, Expression] = params
        self.undefined_args: set = uargs

    def __str__(self) -> str:
        dist = self.dist_name
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        dimshape = tuple([f'{d}={s}' for d, s in zip(self.dims, self.shape)])
        return f"{dist}({params}, dims={dimshape})"
    
    def __repr__(self) -> str:
        return str(self)

    @property
    def dist_name(self) -> str:
        return self._dist_str

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
            dim_shapes=self.simulation.parameter_shapes
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

    # def get_dim_shape(self, dims: Tuple[str, ...]) -> Tuple[int, ...]:
    #     dim_shape = []
    #     for dim in dims:
    #         self.simulation.parameter_shapes
    #         coords = set(self.simulation.observations[dim].values.tolist())
    #         n_coords = len(coords)
    #         dim_shape.append(n_coords)

    #     if len(dim_shape) == 0:
    #         dim_shape = (1,)
    #     return tuple(dim_shape)

    @classmethod
    def parse_model_priors(
        cls, 
        parameters: Dict[str,Param], 
        dim_shapes: Dict[str,Tuple[int, ...]]
    ):
        priors = {}
        for key, par in parameters.items():
            if par.prior is None:
                raise AttributeError(
                    f"No prior was defined for {par}. E.g.: "+
                    f"sim.config.model_parameters.{par} = lognorm(loc=1, scale=2)"
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
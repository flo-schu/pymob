import re
from typing import (
    Dict,
    Tuple,
    Literal,
    Union,
    Callable,
    Any
)
from abc import ABC, abstractmethod

import arviz as az

from pymob.simulation import SimulationBase
from pymob.sim.parameters import Param, RandomVariable, Expression
from pymob.sim.config import Modelparameters

class Distribution:
    distribution_map: Dict[str,Tuple[Callable, Dict[str,str]]] = {}
    """The distribution is a pre-initialized distibution with human friendly 
    interface to construct a distribution in an arbitrary backend.

    The necessary adjustments to make the distribution backend specific
    are done by passing more context to Distribution class to the
    _context variable
    """
    _context = {}
    def __init__(self, name: str, random_variable: RandomVariable) -> None:
        self.name = name
        self._dist_str = random_variable.distribution
        self._parameter_expression = random_variable.parameters
        self._dims = random_variable.dims

        dist, params, uargs = self.parse_distribution(random_variable)
        self.distribution: Callable = dist
        self.parameters: Dict[str, Expression] = params
        self.undefined_args: set = uargs

    def __str__(self) -> str:
        dist = self.distribution.__name__
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{dist}({params})"
    
    def __repr__(self) -> str:
        return str(self)
    
    def construct(self, lookup: Callable):
        context = {arg: lookup(arg) for arg in self.undefined_args}
        context.update(self._context)
        # evaluate the parameters given a context
        params = {
            key: value.evaluate(context=context) 
            for key, value in self.parameters.items()
        }
        return self.distribution(**params)


    @classmethod
    def parse_distribution(cls, random_variable: RandomVariable) -> Tuple[Any,Dict[str,Expression],set]:

        distribution_mapping = cls.distribution_map[random_variable.distribution]

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
            # parsed_val = val.evaluate()
            # parsed_val = cls.generate_transform(expression=val)
            mapped_params.update({mapped_key:val})

        return distribution, mapped_params, underfined_args

    @staticmethod
    @abstractmethod
    def generate_transform(expression: Expression):
        """This function translates parameter transformations into expressions
        that can be used by the inference backend.
        """
        pass



class InferenceBackend(ABC):
    _distribution = Distribution
    idata: az.InferenceData

    def __init__(
        self, 
        simulation: SimulationBase,
    ) -> None:
        self.EPS = 1e-8
        
        self.simulation = simulation
        self.config = simulation.config

        # parse model components
        self.prior = self.parse_model_priors(
            parameters=self.config.model_parameters.free,
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

    @staticmethod
    @abstractmethod
    def generate_transform(expression: Expression):
        """This function translates parameter transformations into expressions
        that can be used by the inference backend.
        """
        pass

    @property
    @abstractmethod
    def distribution_map(self) -> Dict[str,Tuple]:
        pass

    @property
    def extra_vars(self):
        return self.config.inference.extra_vars
    
    @property
    def n_predictions(self):
        return self.config.inference.n_predictions
    
    @classmethod
    def parse_random_variable(cls, parname: str, random_variable: RandomVariable, distribution_map: Dict[str,Tuple]) -> Tuple[str,Any,Dict[str,Callable]]:

        distribution_mapping = distribution_map[random_variable.distribution]

        if not isinstance(distribution_mapping, tuple):
            distribution = distribution_mapping
            distribution_mapping = (distribution, {})
        
        assert len(distribution_mapping) == 2, (
            "distribution and parameter mapping must be "
            "a tuple of length 2."
        )

        distribution, parameter_mapping = distribution_mapping
        mapped_params = {}
        for key, val in random_variable.parameters.items():
            mapped_key = parameter_mapping.get(key, key)
            parsed_val = cls.generate_transform(expression=val)
            mapped_params.update({mapped_key:parsed_val})

        return parname, distribution, mapped_params


    @classmethod
    def parse_model_priors(
        cls, 
        parameters: Dict[str,Param], 
    ):
        priors = {}
        for key, par in parameters.items():
            if par.prior is None:
                raise AttributeError(
                    f"No prior was defined for {par}. E.g.: "+
                    f"sim.config.model_parameters.{par} = lognorm(loc=1, scale=2)"
                )

            dist = cls._distribution(name=key, random_variable=par.prior)
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
                random_variable=error_distribution
            )
               
            error_model.update({data_var: error_dist})
        return error_model
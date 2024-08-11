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

class InferenceBackend(ABC):
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
            distribution_map=self.distribution_map,
        )
        self.evaluator = self.parse_deterministic_model()
        self.error_model = self.parse_error_model()
        
        # combine the model
        self.inference_model = self.parse_probabilistic_model()

    @abstractmethod
    def parse_model_priors(
        self, 
        parameters: Dict[str,Param], 
        distribution_map: Dict[str,Callable]
    ):
        pass

    @abstractmethod
    def parse_deterministic_model(self):
        pass

    @abstractmethod
    def parse_error_model(self):
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
    def distribution_map(self) -> Dict[str,Callable]:
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

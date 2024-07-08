from typing import Callable, Dict, List, Optional, Sequence, Literal, Tuple
from frozendict import frozendict
from dataclasses import dataclass, field
import inspect

@dataclass(frozen=True)
class SolverBase:
    """
    The idea of creating a solver as a class is that it is easier
    to pass on important arguments of the simulation relevant to the 
    Solver. Therefore a solver can access all attributes of an Evaluator
    """
    model: Callable
    dimensions: Tuple
    n_ode_states: int
    coordinates: frozendict[str, Tuple]
    data_variables: Tuple
    is_stochastic: bool
    solver_kwargs: frozendict = frozendict()
    indices: frozendict[str, Tuple] = frozendict()
    post_processing: Optional[Callable] = None

    x_dim: str = "time"
    batch_dimension: str = "batch_id"

    # fields that are computed post_init
    x: Tuple = field(init=False)

    def __post_init__(self, *args, **kwargs):
        object.__setattr__(self, "x", tuple(self.coordinates[self.x_dim]))

    def __call__(self, **kwargs):
        return self.solve(**kwargs)
    
    def solve(self):
        raise NotImplementedError("Solver must implement a solve method.")

def mappar(func, parameters, exclude=[], to:Literal["tuple","dict"]="tuple"):
    func_signature = inspect.signature(func).parameters.keys()
    model_param_signature = [p for p in func_signature if p not in exclude]
    if to == "tuple":
        model_args = [parameters.get(k) for k in model_param_signature]
        model_args = tuple(model_args)
    elif to == "dict":
        model_args = {k: parameters.get(k) for k in model_param_signature}
    else:
        raise NotImplementedError(f"'to={to}' is not implemented for 'mappar'")

    return model_args


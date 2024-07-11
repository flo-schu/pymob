import numpy as np
import xarray as xr
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
    coordinates: frozendict[str, Tuple] = field(repr=False)
    coordinates_input_vars: frozendict[str, frozendict]
    data_variables: Tuple
    is_stochastic: bool
    post_processing: Callable
    solver_kwargs: frozendict = frozendict()
    indices: frozendict[str, Tuple] = field(repr=False, default=frozendict())

    x_dim: str = "time"
    batch_dimension: str = "batch_id"
    extra_attributes = []

    # fields that are computed post_init
    x: Tuple[float] = field(init=False, repr=False)

    def __post_init__(self, *args, **kwargs):
        x = self.coordinates[self.x_dim]
        if not np.all(x[:-1] <= x[1:]):
            raise ValueError(
                f"x_dim '{self.x_dim}' must be sorted in ascending order."
            )
        object.__setattr__(self, "x", x)

        # set extra attributes from solver_kwargs, which are specified through
        # the dispatch_constructor. Those don't receive post-processing
        for key in self.extra_attributes:
            value = self.solver_kwargs.get(key, None)
            if value is not None:
                object.__setattr__(self, key, value)

        self.test_matching_batch_dims()
        

    def __call__(self, **kwargs):
        return self.solve(**kwargs)
    
    def test_matching_batch_dims(self):
        bc = self.coordinates.get(self.batch_dimension, None)

        if bc is not None:
            matching_batch_coords_if_present = [
                v[self.batch_dimension] == bc 
                for k, v in self.coordinates_input_vars.items() 
                if self.batch_dimension in v
            ]

            if not all(matching_batch_coords_if_present):
                raise IndexError(
                    f"Batch coordinates '{self.batch_dimension}' of input "
                    "variables do not have the same size "
                    "as the batch dimension of the observations."
                )

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


def create_interpolation(
        x_in: xr.Dataset, 
        x: str="time", 
        factor: float=1e-4, 
        interpolation: Literal["fill-forward", "linear"] = "fill-forward",
    ) -> xr.Dataset:
    """Make the interpolation safe by adding a coordinate just before each 
    x-value (except the first vaue). The distance between the new and the next
    point are calculated as a fraction of the previous distance between
    neighboring points. The corresponding y-values are first set to NaN and then
    interpolated based on the interpolation method.

    Parameters
    ----------
    x_in : xr.Dataset
        The input dataset which contains a coordinate (x) and a data variable
        (y)
    x : str, optional
        The name of the x coordinate, by default "time"
    factor : float, optional
        The distance between the newly added points and the following existing
        points on the x-scale, by default 1e-4
    interpolation : Literal["fill-forward", "linear"], optional
        The interpolation method. In addition to 'fill-forward' and 'linear',
        any method give in `xr.interpolate_na` can be chosen, by default
        "fill-forward"

    Returns
    -------
    xr.Dataset
        The interpolated dataset
    """
    xs = x_in.coords[x]

    # calculate x values that are located just a little bit smaller than the xs
    # where "just a little bit" is defined by the distance to the previous x
    # and a factor. This way the scale of the observations should not matter
    # and even very differently sized x-steps should be interpolated correctly
    fraction_before_xs = (
        xs.isel({x:range(1, len(xs))}).values
        - xs.diff(dim=x) * factor
    )

    # create a sorted time vector
    xs = sorted([*fraction_before_xs.values, *xs.values])

    # add new time indices with NaN values 
    x_in_reindexed = x_in.reindex({x:xs})

    if interpolation == "fill-forward":
        # then fill nan values with the previous value (forward-fill)
        x_in_interpolated = x_in_reindexed.ffill(dim=x, limit=1)

    else:
        x_in_interpolated = x_in_reindexed.interpolate_na(dim=x, method="linear")

    return x_in_interpolated

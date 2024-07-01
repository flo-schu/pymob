from typing import Literal
import inspect
import xarray as xr
from scipy.integrate import solve_ivp

class SolverBase:
    """
    The idea of creating a solver as a class is that it is easier
    to pass on important arguments of the simulation relevant to the 
    Solver. Therefore a solver can access all attributes of an Evaluator
    """
    def __call__(self, **kwargs):
        return self.solve(**kwargs)
    
    @staticmethod
    def solve():
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
        y: str, 
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



def solve_analytic_1d(model, parameters, dimensions, coordinates, data_variables, seed=None):
    """Solves an anlytic function for all coordinates in the first data 
    dimension of the model

    - parameters: define the model
    - y0: set the initial values of the ODE states
    - coordinates: are needed to know over which values to integrate
    - seed: In case stochastic processes take place inside the model this is necessary
    
    In order to make things explicit, all information which is needed by the
    model needs to be specified in the function signature. 
    This also makes the solvers functionally oriented, a feature that helps the
    usability of models accross inference frameworks. Where functions should not
    have side effects.

    Additionally, passing arguments via the signature makes it easier to write
    up models in a casual way and only later embed them into more regulated
    structures such as pymob

    """
    
    model_args = mappar(
        model, 
        parameters["parameters"], 
        exclude=["t", "x"] + dimensions,
        to="dict"
    )
    
    x = coordinates[dimensions[0]]

    model_results = []
    results = model(x, **model_args)
    model_results.append(results)
    
    return {data_var:y for data_var, y in zip(data_variables, model_results)}


def solve_ivp_1d(model, parameters, coordinates, data_variables):
    """Initial value problems always need the same number of recurrent arguments

    - parameters: define the model
    - y0: set the initial values of the ODE states
    - coordinates: are needed to know over which values to integrate
    - seed: In case stochastic processes take place inside the model this is necessary
    
    In order to make things explicit, all information which is needed by the
    model needs to be specified in the function signature. 
    This also makes the solvers functionally oriented, a feature that helps the
    usability of models accross inference frameworks. Where functions should not
    have side effects.

    Additionally, passing arguments via the signature makes it easier to write
    up models in a casual way and only later embed them into more regulated
    structures such as pymob

    """
    odeargs = mappar(model, parameters["parameters"], exclude=["t", "y"])
    t = coordinates["time"]
    results = solve_ivp(
        fun=model,
        y0=parameters["y0"],
        t_span=(t[0], t[-1]),
        t_eval=t,
        args=odeargs,
    )
    return {data_var:y for data_var, y in zip(data_variables, results.y)}

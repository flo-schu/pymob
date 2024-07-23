import numpy as np
import xarray as xr
from typing import Callable, Dict, List, Optional, Sequence, Literal, Tuple
from frozendict import frozendict
from dataclasses import dataclass, field
import inspect
from scipy.ndimage import gaussian_filter1d
from diffrax import rectilinear_interpolation

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
    extra_attributes = ["exclude_kwargs_model", "exclude_kwargs_postprocessing"]
    exclude_kwargs_model = ["t", "x_in", "y", "X"]
    exclude_kwargs_postprocessing = ["t", "time", "interpolation", "results"]

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
        self.test_x_coordinates()
        

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

    def test_x_coordinates(self):
        x = self.coordinates[self.x_dim]
        if "x_in" not in self.coordinates_input_vars:
            return
        
        x_xin = self.coordinates_input_vars["x_in"][self.x_dim]

        if np.max(x) > np.max(x_xin):
            raise AssertionError(
                f"The {self.x_dim}-coordinate on the observations (sim.coordinates) "
                f"goes to a higher {self.x_dim} than the {self.x_dim}-coordinate "
                "of the model_parameters['x_in']. "
                "Make sure to run the simulation only until the provided x_in "
                "values, or extend the x_in values until the required time"
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


def jump_interpolation(
        x_in: xr.Dataset, 
        x_dim: str="time", 
        factor: float=0.001, 
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
    x_dim : str, optional
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
    xs = x_in.coords[x_dim]

    # calculate x values that are located just a little bit smaller than the xs
    # where "just a little bit" is defined by the distance to the previous x
    # and a factor. This way the scale of the observations should not matter
    # and even very differently sized x-steps should be interpolated correctly
    fraction_before_xs = (
        xs.isel({x_dim:range(1, len(xs))})
        - xs.diff(dim=x_dim) * factor
    )

    # create a sorted time vector
    xs = sorted([*fraction_before_xs.values, *xs.values])

    # add new time indices with NaN values 
    x_in_reindexed = x_in.reindex({x_dim:xs})

    if interpolation == "fill-forward":
        # then fill nan values with the previous value (forward-fill)
        x_in_interpolated = x_in_reindexed.ffill(dim=x_dim, limit=1)

    else:
        x_in_interpolated = x_in_reindexed.interpolate_na(dim=x_dim, method="linear")

    return x_in_interpolated


def smoothed_interpolation(
    x_in: xr.Dataset, 
    x_dim: str="time", 
    factor: float=0.001, 
    sigma: int = 20,
) -> xr.Dataset:
    """Smooth the interpolation by first creating a dense x vector and forward
    filling all ys. Following this the values are smoothed by a gaussian filter.

    Parameters
    ----------
    x_in : xr.Dataset
        The input dataset which contains a coordinate (x) and a data variable
        (y)
    x_dim : str, optional
        The name of the x coordinate, by default "time"
    factor : float, optional
        The distance between the newly added points and the following existing
        points on the x-scale, by default 1e-4

    Returns
    -------
    xr.Dataset
        The interpolated dataset
    """
    xs = x_in.coords[x_dim]
    assert factor > 0, "Factor must be larger than zero, to ensure correct ordering"
    
    xs_extra = np.arange(xs.values.min(), xs.values.max()+factor, step=factor)
    xs_ = np.sort(np.unique(np.concatenate([xs.values, xs_extra])))

    # add new time indices with NaN values 
    x_in_reindexed = x_in.reindex({x_dim:xs_})

    # then fill nan values with the previous value (forward-fill)
    x_in_interpolated = x_in_reindexed.ffill(dim=x_dim)

    # Apply Gaussian smoothing
    sigma = 20  # Adjust sigma for desired smoothness
    for k in x_in_interpolated.data_vars.keys():
        y = x_in_interpolated[k]
        y_smoothed = gaussian_filter1d(y.values, sigma, axis=list(y.dims).index(x_dim))

        x_in_interpolated[k].values = y_smoothed

    return x_in_interpolated


def radius_interpolation(
    x_in: xr.Dataset, 
    x_dim: str="time", 
    radius: float=0.1, 
    num_points: int=10,
    rectify=True
) -> xr.Dataset:
    """Smooth the interpolation by first creating a dense x vector and forward
    filling all ys. Following this the values are smoothed by a gaussian filter.

    WARNING! It is very pretty but does not work with diffrax

    Parameters
    ----------
    x_in : xr.Dataset
        The input dataset which contains a coordinate (x) and a data variable
        (y)
    x_dim : str, optional
        The name of the x coordinate, by default "time"
    radius : float, optional
        The radius of the quarter-circle to curve the jump transition. 
        By default 0.1
    num_points : int, optional
        The number of points to interpolate each jump with. Default: 10
    rectify : bool
        Whether the input should be converted to a stepwise pattern. Default 
        is True. This is typically applied if an unprocessed signal is included.
        E.g. the signal was observed y_i 
        
    Returns
    -------
    xr.Dataset
        The interpolated dataset
    """
    x = x_in.coords[x_dim] 
    assert radius <= np.diff(np.unique(x)).min() / 2

    if rectify:
        x = np.concatenate([[x[0]], *[[x_i-0.00, x_i] for x_i in x[1:]]])

    data_arrays = []
    for k in x_in.data_vars.keys():
        y = x_in[k]
        if rectify:
            yvals = np.concatenate([*[[y_i, y_i] for y_i in y[:-1]], [y[-1]]])
        else:
            yvals = y.values

        x_interpolated = [np.array(x[0],ndmin=1)]
        y_interpolated = [np.array(yvals[0], ndmin=2)]
        for i in range(0, len(x) - 1):
            x_, y_, = curve_jumps(x, yvals, i, r=radius, n=num_points)

            x_interpolated.append(x_)
            y_interpolated.append(y_)
    
        x_interpolated = np.concatenate(x_interpolated)
        x_uniques = np.where(np.diff(x_interpolated) != 0)
        x_interpolated = x_interpolated[x_uniques]

        y_interpolated = np.row_stack(y_interpolated)[x_uniques]
        
        coords = {x_dim: x_interpolated}
        coords.update({d: y.coords[d].values for d in y.dims if d != x_dim})

        y_reindexed = xr.DataArray(
            y_interpolated, 
            coords=coords,
            name=y.name
        )
        data_arrays.append(y_reindexed)

    x_in_interpolated = xr.combine_by_coords(data_objects=data_arrays)
    
    # there will be nans if the data variables have different steps
    # x_in_interpolated = x_in_interpolated.interpolate_na(
    #     dim="time", method="linear"
    # )

    return x_in_interpolated # type: ignore


def curve_jumps(x, y, i, r, n):
    x_i = x[i] # jump start
    y_i = y[i] # jump start
    y_im1 = y[i-1] # jump start
    y_ip1 = y[i+1] # jump end

    def circle_func(x, r, a): 
        # using different radii does not work, because this would also require different x_values
        arc = r**2 - (x - a)**2
        return np.sqrt(np.where(arc >= 0, arc, 0))
    
    # end of jump
    dy_im1 = y_i - y_im1 # jump difference to previous point
    if np.all(dy_im1 == 0):
        dyj = y_ip1 - y_i
        sj = np.where(np.abs(dyj) > r, np.sign(dyj), dyj / 2 / r) # direction and size of the jump, scaled by the radius
        # sj = np.clip((y_ip1 - y_i)/r/2, -1, 1) # direction and size of the jump, scaled by the radius
        xc = np.linspace(x_i-r, x_i-r*0.001, num=n)
        yc = y_i + (np.einsum("k,x->xk",  -sj, circle_func(x=xc, r=r, a=x_i-r)) + sj * r)
    else:
        dyj = y_i - y_im1
        sj = np.where(np.abs(dyj) > r, np.sign(dyj), dyj / 2 / r) # direction and size of the jump, scaled by the radius
        xc = np.linspace(x_i+ r*0.001, x_i+r, num=n)
        yc = y_i + (np.einsum("k,x->xk",  sj, circle_func(x=xc, r=r, a=x_i+r)) - sj * r)
    
    return xc, yc


def rect_interpolation(
    x_in: xr.Dataset, 
    x_dim: str="time", 
):  
    """Use diffrax' rectilinear_interpolation. To add values and interpolate
    one more step after the end of the timeseries
    """
    data_arrays = []
    for k in x_in.data_vars.keys():
        v = x_in[k]
        x = v[x_dim].values
        y = v.values
        xs, ys = rectilinear_interpolation(
            ts=np.concatenate([x, np.array(x[-1]+1, ndmin=1)]), # type:ignore
            ys=np.row_stack([y, np.array(y[-1], ndmin=2)]) # type: ignore
        )

        coords = {x_dim: xs}
        coords.update({d: v.coords[d].values for d in v.dims if d != x_dim})

        y_reindexed = xr.DataArray(
            ys, 
            coords=coords,
            name=v.name
        )
        data_arrays.append(y_reindexed)

    x_in_interpolated = xr.combine_by_coords(data_objects=data_arrays)

    return x_in_interpolated
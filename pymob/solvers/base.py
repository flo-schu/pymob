import numpy
import numpy as np
from numpy.typing import ArrayLike
from types import ModuleType
import xarray as xr
from typing import (
    Callable, Dict, List, Optional, Sequence, Literal, Tuple, Union
)
from frozendict import frozendict
from dataclasses import dataclass, field
import inspect
import functools
import warnings
from scipy.ndimage import gaussian_filter1d
from diffrax import rectilinear_interpolation
from pymob.utils.errors import PymobError

@dataclass(frozen=True)
class SolverBase:
    """
    The idea of creating a solver as a class is that it is easier
    to pass on important arguments of the simulation relevant to the 
    Solver. Therefore a solver can access all attributes of an Evaluator
    """
    model: Callable
    dimensions: Tuple
    dimension_sizes: frozendict[str, int]
    parameter_dims: frozendict[str, Tuple[str, ...]]
    n_ode_states: int
    coordinates: frozendict[str, Tuple] = field(repr=False)
    coordinates_input_vars: frozendict[str, frozendict[str, frozendict[str, Tuple[Union[float,int,str], ...]]]]
    dims_input_vars: frozendict[str, frozendict[str, Tuple[str, ...]]]
    coordinates_indices: frozendict[str, tuple]
    data_variables: Tuple
    data_structure_and_dimensionality: frozendict[str, frozendict[str, int]]
    is_stochastic: bool
    post_processing: Callable

    # optional
    solver_kwargs: frozendict = frozendict()
    indices: frozendict[str, Tuple] = field(repr=False, default=frozendict())

    x_dim: str = "time"
    batch_dimension: str = "batch_id"
    # the defaults of pymob.sim.config.sections.Solverbase must reflect these
    exclude_kwargs_model: Tuple[str, ...] = ("t", "y", "x_in",)
    # post_processing is validated in __post_init__ and always has the
    # signature f(t, y, x_in, **params) (legacy functions are wrapped)
    exclude_kwargs_postprocessing: Tuple[str, ...] = ("t", "y", "x_in",)

    # fields that are computed post_init
    x: Tuple[float] = field(init=False, repr=False)
    shapes_coordinates: Dict[str, int] = field(init=False, repr=False)
    shapes_parameter_coordinates: Dict[str, Tuple[int, ...]] = field(init=False, repr=False)
    shapes_input_vars_coordinates: frozendict[str, frozendict[str, Tuple[int, ...]]] = field(init=False, repr=False)
    len_batch_coordinate: int = field(init=False)
    x_shape_batched: Tuple[int, ...] = field(init=False)

    def __post_init__(self, *args, **kwargs):
        # translate legacy argument names (results, time, interpolation) in
        # exclude_kwargs_postprocessing to the current names (y, t, x_in)
        exclude_kwargs_postprocessing = self._translate_legacy_exclude_kwargs_postprocessing(
            self.exclude_kwargs_postprocessing
        )
        object.__setattr__(
            self, "exclude_kwargs_postprocessing", exclude_kwargs_postprocessing
        )

        # make sure the model follows the signature f(t, y, [x_in], *params)
        self.validate_model(self.model)

        # make sure post processing follows the signature f(t, y, x_in, **params)
        # legacy functions f(results, time, interpolation, *params) are wrapped
        post_processing = self.validate_post_processing(self.post_processing)
        object.__setattr__(self, "post_processing", post_processing)

        x = self.coordinates[self.x_dim]
        if not np.all(x[:-1] <= x[1:]):
            raise ValueError(
                f"x_dim '{self.x_dim}' must be sorted in ascending order."
            )
        object.__setattr__(self, "x", x)

        coord_dims = self._get_coordinate_unique_shapes()
        object.__setattr__(self, "shapes_coordinates", coord_dims)
        
        parameter_shapes = self._get_parameter_shapes()
        object.__setattr__(self, "shapes_parameter_coordinates", parameter_shapes)

        input_vars_shapes = self._get_input_vars_shapes()
        object.__setattr__(self, "shapes_input_vars_coordinates", input_vars_shapes)

        batch_length = self._get_batch_length()
        object.__setattr__(self, "len_batch_coordinate", batch_length)

        x_shape_batched = self._get_x_shape_batched()
        object.__setattr__(self, "x_shape_batched", x_shape_batched)

        # set extra attributes from solver_kwargs, which are specified through
        # the dispatch_constructor. Those don't receive post-processing

        self.test_matching_batch_dims()
        self.test_x_coordinates()
        

    def __call__(self, **kwargs):
        return self.solve(**kwargs)
    
    def _get_batch_length(self) -> int:
        batch_coordinates = self.coordinates.get(self.batch_dimension, [0])
        return len(batch_coordinates)

    def _get_x_shape_batched(self) -> Tuple[int, ...]:
        n_batch = self.len_batch_coordinate
        n_x = self.dimension_sizes[self.x_dim]
        return (n_batch, n_x)

    def _get_coordinate_unique_shapes(self, ) -> frozendict[str, int]:
        coordinate_shape_dict = {}
        for key, coords in self.coordinates.items():
            n_coords = len(coords)
            coordinate_shape_dict.update({key: n_coords})

        for key, coords in self.coordinates_indices.items():
            unique_coords = set(coords)
            n_coords = len(unique_coords)
            coordinate_shape_dict.update({key: n_coords})

        return frozendict(coordinate_shape_dict)

    def _get_parameter_shapes(self, ) -> frozendict[str, Tuple[int, ...]]:
        par_shape_dict = {}
        for par_name, par_dims in self.parameter_dims.items():
            dim_shape = []
            for d in par_dims:
                if d == self.batch_dimension and d not in self.dimension_sizes:
                    dim_size = 1
                else:
                    try:
                        dim_size = self.dimension_sizes[d]
                    except KeyError as err:
                        raise KeyError(
                            f"KeyError: Dimension '{d}' could not be found in "+
                            f"any dimension specified for the simulation "+
                            f"{self.dimension_sizes}. It must be either "+
                            "specified via the `sim.indices` dict."
                        )
                
                dim_shape.append(dim_size)
            par_shape_dict.update({par_name: tuple(dim_shape)})

        return frozendict(par_shape_dict)

    def _get_input_vars_shapes(self, ) -> frozendict[str, frozendict[str, Tuple[int, ...]]]:
        input_vars_shape_dict = {}
        for input_var_name, input_var_dict in self.dims_input_vars.items():
            data_var_shape_dict = {}
            for data_var_name, data_var_dims in input_var_dict.items():
                dim_shape = []
                for d in data_var_dims:
                    # look for the dimension in the coodinates of the input vars
                    coords_data_var = self.coordinates_input_vars[input_var_name][data_var_name]
                    coords = coords_data_var.get(d, None)
                    
                    # if it is found use its length as a dimension size
                    if coords is not None:
                        dim_size = len(coords)
                    
                    # if it is not found, see if the dim is the batch dimensions
                    # and handle the batch dimension size
                    # this is done, in cases where there is a batch dimension, 
                    # but it was not specified for the input, because it is 
                    # expected that it simply broadcasts to the batch dimension
                    elif d == self.batch_dimension:
                        if d not in self.dimension_sizes:
                            dim_size = 1
                        else:
                            dim_size = self.dimension_sizes[d]

                    # if the dimension is not in the coordinates of the input vars
                    # or not as a batch dimension in the dimension sizes dict
                    # something is severely wrong
                    else:
                        raise KeyError(
                            f"KeyError: Dimension: '{d}' for data_var: "+
                            f"'{data_var_name}' in input var: '{input_var_name}' "+
                            f"could not be found in the coordinate dimensions: "+
                            f"{list(coords_data_var.keys())}. "+
                            f"Make sure `sim.model_parameters['{input_var_name}']` "+
                            "is correctly specified. It is recommended to "+
                            "specify the model_input with `sim.config.simulation."+
                            f"{input_var_name} = ['...', '{data_var_name}=...', '...']` "+
                            f"and use `sim.parse_input(input='{input_var_name}', "+
                            f"reference_data=..., drop_dims=[...])` to parse the "+
                            "input."
                        )
                    
                    dim_shape.append(dim_size)
                data_var_shape_dict.update({data_var_name: tuple(dim_shape)})
            input_vars_shape_dict.update({
                input_var_name: frozendict(data_var_shape_dict)
            })
        return frozendict(input_vars_shape_dict)

    def test_matching_batch_dims(self):
        bc = self.coordinates.get(self.batch_dimension, None)

        if bc is not None:
            for input_key, input_dict in self.coordinates_input_vars.items():
                matching_batch_coords_if_present = {}
                for k, v in input_dict.items():
                    if self.batch_dimension in v:
                        matching_batch_coords_if_present.update({k:  
                            v[self.batch_dimension] == bc 
                        })

                if not all(list(matching_batch_coords_if_present.values())):
                    raise PymobError(
                        f"The batch coordinates of the '{input_key}' input variable "+
                        f"'{k}': {dict(v)} differ from the "+
                        f"batch coordinates of the observations {{'{self.batch_dimension}': {bc}.}} "+
                        "\n\n" +

                        "Why does this error occur?\n" +
                        "--------------------------\n" +
                        "Pymob internally requires that all model inputs (theta, x_in, y0) to have "+ 
                        "equally sized batch dimensions and broadcasts parameters and input "+
                        "according to the respective batch dimensions, if they are not "+
                        "provided in the expanded form. "+
                        "Differing coordinates would lead to inhomogeneous batch dimensions, "+
                        "which cannot be processed by the solver. "+
                        f"You have possibly set `sim.model_parameters['{input_key}']` " +
                        f"with an xarray.Dataset that has different {self.batch_dimension}"+
                        "-coordinates than `sim.observations`."
                        "\n\n"

                        "How to fix this error?\n"+
                        "----------------------\n"+
                        "Make sure all model inputs have the same batch coordinates. "+
                        "Try using the pymob method `SimulationBase.parse_input(...) -> "+
                        "https://pymob.readthedocs.io/en/stable/api/pymob.html#pymob.simulation.SimulationBase.parse_input "
                        "to prepare 'x_in' and 'y0'. E.g.: \n"+
                        "* `sim.model_parameters['y0'] = sim.parse_input('y0', drop_dims=['time'])`\n"+
                        "* `sim.model_parameters['x_in'] = sim.parse_input('x_in', reference_data=sim.observations)` "
                    )

    def test_x_coordinates(self):
        x = self.coordinates[self.x_dim]
        if len(self.coordinates_input_vars["x_in"]) == 0:
            return
        
        x_xin = [
            np.max(v.get(self.x_dim, [0])) 
            for v in self.coordinates_input_vars["x_in"].values()
        ]

        if np.max(x) > np.max(x_xin):
            raise AssertionError(
                f"The {self.x_dim}-coordinate on the observations (sim.coordinates) "+
                f"goes to a higher {self.x_dim} than the {self.x_dim}-coordinate "+
                f"of the model_parameters['x_in'] ({np.max(x)} > {np.max(x_xin)}). "+
                "Make sure to run the simulation only until the provided x_in "+
                "values, or extend the x_in values until the required time"
            )

    def preprocess_parameters(self, parameters, num_backend: ModuleType=numpy):
        ode_args = mappar(
            self.model, 
            parameters, 
            exclude=self.exclude_kwargs_model, 
            to="dict"
        )
        ode_args_broadcasted = self._broadcast_args(
            arg_dict=frozendict(ode_args), # type: ignore
            num_backend=num_backend
        )
        
        pp_args = mappar(
            self.post_processing, 
            parameters, 
            exclude=self.exclude_kwargs_postprocessing, 
            to="dict"
        )
        pp_args_broadcasted = self._broadcast_args(
            arg_dict=frozendict(pp_args), # type: ignore
            num_backend=num_backend
        )

        return ode_args_broadcasted, pp_args_broadcasted

    def test_batch_dim_consistency(self, X_in, Y_0, ode_args, pp_args, num_backend: ModuleType=numpy):
        # This method is currently not used. It may come in handy later on, but it 
        # would need to be called during Evaluator.__call__(), which creates unnecessary
        # overhead calculations, which would slow down pymob. Instead the check is done
        # during the dispatch_constructor() call. Specifically in 
        # SolverBase.test_matching_batch_dims
        
        shapes = {
            # xin data shape
            "theta":  [oa.shape[0] for oa in ode_args] + [pa.shape[0] for pa in pp_args],
            "y0":     [y0.shape[0] for y0 in Y_0],
            "x_in":   [xin[0].shape[0] for xin in X_in],
        }

        _shapes = numpy.concatenate(list(shapes.values()))

        if not all(x == _shapes[0] for x in _shapes):
            raise PymobError(
                f"The sizes of the batch dimensions ('{self.batch_dimension}') of theta, " +
                f"x_in and y0 did not match: {shapes}. This problem is often caused if " +
                "the components of sim.model_parameters have not been harmonized, "
            )


    def _broadcast_args(self, arg_dict: frozendict[str, numpy.ndarray], num_backend: ModuleType=numpy):
        # simply broadcast the parameters along the batch dimension
        # if there is no other index provided
        n_batch = self.len_batch_coordinate

        args = []
        for arg_name, arg in arg_dict.items():
            # you can expect that any of the expected parameter_shapes have the
            # size of the batch_dimension in the first axis.
            # Note that this is the taget shape without any extended dimension.
            # Dimensions are only extended, if there is no batch dimension already
            # present
            target_shape = self.shapes_parameter_coordinates.get(arg_name, (n_batch, ))

            # make sure the argument is an array 
            arg_promoted = num_backend.array(arg, dtype=float)

            # if the dimensionality of the arg is zero, then it is a scalar and should
            # not be an array. It should consequently be only broadcasted to the target
            # shape which should not raise any broadcasting errors
            if arg_promoted.ndim == 0:
                if arg_name in self.shapes_parameter_coordinates:
                    arg_broadcasted = num_backend.broadcast_to(
                        arg_promoted, shape=self.shapes_parameter_coordinates[arg_name]
                    )
                else:
                    # if there is no broadcasting shape, assume the batch dimension
                    # is one and there is no alternative broadcasting necessary
                    arg_broadcasted = num_backend.broadcast_to(
                        arg_promoted, shape=(1,)
                    )

            # if the size of the 1st dimension of the argument array 
            # is identical to the size of the specified first dimension
            # we know that the input array was constructed to match the size.
            # The length of the batch coordinates is at least 1
            # There are weird cases in which the input arrays are square,
            # and the dimensions of the parameter have been specified in a 
            # different order (other than batch_dimension first)
            # This problem has been fixed, as the dimensional order is now
            # checked in the Evaluator.__init__
            elif arg_promoted.shape[0] == target_shape[0]:
                # if greater zero (zero dim not possible because of the 
                # promotion to 1D arrays at the beginning of the loop)
                # then it is assumed that the array correctly contains 
                # more than one value for each id in the batch dimension
                # i.e. vector, matrix or nd-array parameters in the ODE
                # We leave the array as is
                arg_broadcasted = arg_promoted

            elif (
                # this is when passed and expected shapes have the same number
                # of dimensions
                arg_promoted.shape[0] == 1 
                # this is when the shape of the later dimensions match.
                # I.e. the batch_dimension is not in the passed argument values
                # This happens frequently, when parameters are not broadcasted
                # to the batch dimension before being passed to the solver
                #
                # An example for the syntax below:
                # arg_promoted.shape >>> (2,5) 
                # target_shape >>> (10,2,5)
                # len(arg_promoted.shape) >>> 2
                # target_shape[-2:] >>> (2,5)
                # (2,5) == (2,5) >>> True
                or arg_promoted.shape == target_shape[-len(arg_promoted.shape):]
            ):
                # Note that this will also broadcast parameters across multiple
                # dimension e.g. (5,) -> (10,2,5)
                if arg_promoted.ndim > len(self.shapes_parameter_coordinates[arg_name]):
                    raise ValueError(
                        f"Parameter '{arg_name}' values have shape "+
                        f"{arg_promoted.shape}. This is in conflict with the "+
                        f"specified shape {self.shapes_parameter_coordinates[arg_name]} "+
                        f"from the dimensions {self.parameter_dims[arg_name]}."
                    )
                # the operation will broadcase argument arrays that have the
                # same number of dimensions (and the first dimension is )
                arg_broadcasted = num_backend.broadcast_to(
                    arg_promoted, shape=self.shapes_parameter_coordinates[arg_name]
                )

                # also here we apply broadcasting to the array if the result has 
                # only one dimension
                if arg_broadcasted.ndim == 1:
                    arg_broadcasted = num_backend.expand_dims(arg_broadcasted, -1)
            else:
                raise ValueError(
                    f"The values of parameter '{arg_name}' with the shape "+
                    f"{arg_promoted.shape} could not be broadcasted to the "+
                    f"specified shape {self.shapes_parameter_coordinates[arg_name]} "+
                    f"from the dimensions {self.parameter_dims[arg_name]}. "+
                    f"Make sure you add the missing dimensions in the "+
                    f"parameter specification Param(..., dims=(...,)) "+
                    f"and handle parameter coordinates appropriately. "+
                    "Parameter dimension coordinates can be specified either "+
                    f"1) Add the missing dimension in sim.config.parameters.{arg_name} "+
                    "and sim.coordinates['MISSING_DIM'] = [...]"+
                    "2) Add the missing dimension in "+
                    "sim.config.data_structure.ANY_DATA_VAR and sim.coordinates['MISSING_DIM'] "+
                    "3) dimension in sim.indices "+
                    f"the batch dimension {self.batch_dimension} is added "+
                    f"automatically."
                )
            
            args.append(arg_broadcasted)

        return tuple(args)


    def preprocess_x_in(self, x_in, num_backend:ModuleType=numpy):
        X_in_list = []
        for x_in_var, x_in_vals in x_in.items():
            # parse to array
            x_in_x = num_backend.array(
                self.coordinates_input_vars["x_in"][x_in_var][self.x_dim], 
                dtype=float,
                ndmin=1
            )
            x_in_y = num_backend.array(x_in_vals, ndmin=1, dtype=float)

            # broadcast x
            x_in_batched_shape = (self.len_batch_coordinate, *x_in_x.shape)
            x_in_x_broadcasted = num_backend.broadcast_to(x_in_x, x_in_batched_shape)

            # broadcast y
            y_in_batched_shape = self.shapes_input_vars_coordinates["x_in"][x_in_var]
            x_in_y_broadcasted = num_backend.broadcast_to(x_in_y, y_in_batched_shape) 

            # also here we apply broadcasting to the array if the result has 
            # only one dimension
            if x_in_y_broadcasted.ndim == 1:
                x_in_y_broadcasted = num_backend.expand_dims(
                    x_in_y_broadcasted, -1
                )

            # combine xs and ys to make them ready for interpolation
            X_in = [
                num_backend.array(v) for v in 
                [x_in_x_broadcasted, x_in_y_broadcasted]
            ]

            X_in_list.append(X_in)

        return X_in_list
    
    def preprocess_y_0(self, y0, num_backend:ModuleType=numpy):
        Y0 = []

        for y0_var, y0_vals in y0.items():
            y0_vals_promoted = num_backend.array(y0_vals, ndmin=1, dtype=float)
            
            # wrap y0 data in a dummy batch dim if the batch dim is not
            # included in the coordinates
            y0_var_shape = self.shapes_input_vars_coordinates["y0"][y0_var]
            y0_vals_broadcasted = num_backend.broadcast_to(
                y0_vals_promoted, 
                y0_var_shape
            ) 

            # also here we apply broadcasting to the array if the result has 
            # only one dimension

            Y0.append(y0_vals_broadcasted)
        return Y0


    def solve(self):
        raise NotImplementedError("Solver must implement a solve method.")

    @staticmethod
    def _translate_legacy_exclude_kwargs_postprocessing(
        exclude_kwargs: Sequence[str]
    ) -> Tuple[str, ...]:
        """Translate legacy names in ``exclude_kwargs_postprocessing`` to the
        names of the current post-processing signature f(t, y, x_in, **params)
        (``results -> y``, ``time -> t``, ``interpolation -> x_in``).

        Duplicates are removed. The current arguments are ordered as
        ``(t, y, x_in)`` and followed by any other names in their given order.
        A ``DeprecationWarning`` is issued if legacy names are found.
        """
        exclude_kwargs = tuple(exclude_kwargs)
        legacy_names = [
            k for k in exclude_kwargs if k in LEGACY_POST_PROCESSING_NAME_MAP
        ]
        if len(legacy_names) == 0:
            return exclude_kwargs

        warnings.warn(
            f"exclude_kwargs_postprocessing={exclude_kwargs} contains the legacy "
            f"argument names {legacy_names}. They are translated to the current "
            "post-processing signature f(t, y, x_in, **params) "
            f"({', '.join(f'{k} -> {v}' for k, v in LEGACY_POST_PROCESSING_NAME_MAP.items())}). "
            "Update your configuration to `exclude_kwargs_postprocessing = t y x_in` "
            "in the [solverbase] section (or `sim.config.solverbase."
            "exclude_kwargs_postprocessing = ('t', 'y', 'x_in')`). Support for the "
            "legacy names will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        translated = [LEGACY_POST_PROCESSING_NAME_MAP.get(k, k) for k in exclude_kwargs]
        core = [k for k in ("t", "y", "x_in") if k in translated]
        others = []
        for k in translated:
            if k not in core and k not in others:
                others.append(k)

        return tuple(core + others)

    @staticmethod
    def _check_reserved_parameters(
        func: Callable,
        reserved: Tuple[str, ...],
        n_leading: int,
        func_type: str,
        expected_signature: str,
    ) -> None:
        """Raise an error if any parameter after the ``n_leading`` arguments
        of ``func`` has a name that is listed in ``reserved``."""
        signature = inspect.signature(func)
        parameters = list(signature.parameters.values())[n_leading:]
        clashing = [p.name for p in parameters if p.name in reserved]

        if len(clashing) > 0:
            raise PymobError(
                f"The {func_type} function '{getattr(func, '__name__', func)}"
                f"{signature}' contains the parameter(s) {clashing}, which are "
                f"reserved arguments {reserved}."
                "\n\n"
                "Why does this error occur?\n"
                "--------------------------\n"
                f"{func_type.capitalize()} functions must have the signature "
                f"{expected_signature}. The reserved arguments {reserved} are "
                "passed by the solver and must be the leading arguments of the "
                "function. All following arguments are interpreted as parameters "
                "and are looked up in the model parameters. Reserved arguments "
                "cannot be used as parameter names, because they would not be "
                "passed to the function."
                "\n\n"
                "How to fix this error?\n"
                "----------------------\n"
                f"Make sure that {clashing} are only used as the leading "
                f"arguments of the function {expected_signature}, or rename "
                "the parameter(s)."
            )

    def validate_model(self, func: Callable) -> None:
        """Validate the signature of the model ``f(t, y, [x_in], *params)``.

        The arguments listed in ``exclude_kwargs_model`` are passed by the
        solver. They may only appear as leading arguments of the model.
        A ``PymobError`` is raised if any of the parameters following the
        leading reserved arguments is listed in ``exclude_kwargs_model``.
        """
        if not callable(func):
            raise PymobError(
                f"The model '{func}' (type: {type(func)}) is not callable. "
                "Provide a function with the signature f(t, y, [x_in], *params)."
            )

        reserved = tuple(self.exclude_kwargs_model)
        parameter_names = list(inspect.signature(func).parameters.keys())
        n_leading = 0
        for name in parameter_names:
            if name not in reserved:
                break
            n_leading += 1

        self._check_reserved_parameters(
            func=func,
            reserved=reserved,
            n_leading=n_leading,
            func_type="model",
            expected_signature=f"f({', '.join(reserved)}, *params)",
        )

    def validate_post_processing(self, func: Callable) -> Callable:
        """Validate the signature of a post-processing function.

        The current pymob signature of post-processing functions is
        ``f(t, y, x_in, **params)``, where the leading arguments are defined by
        ``exclude_kwargs_postprocessing``. Functions that contain all of these
        arguments are returned unchanged.

        Legacy post-processing functions with the signature
        ``f(results, time, interpolation, *params)`` are wrapped for backwards
        compatibility. In legacy functions only the positional order of the first
        three arguments was relevant, so they are mapped by position:
        ``results -> y``, ``time -> t``, ``interpolation -> x_in``. The remaining
        parameters are passed on unchanged.

        In both cases a ``PymobError`` is raised if any parameter following the
        leading arguments is listed in ``exclude_kwargs_postprocessing``.

        Parameters
        ----------
        func : Callable
            The post-processing function

        Returns
        -------
        Callable
            A function that can be called as ``f(t, y, x_in, **params)`` and whose
            signature (``inspect.signature``) is ``(t, y, x_in, **params)``.
        """
        if not callable(func):
            raise PymobError(
                f"The post-processing function '{func}' (type: {type(func)}) is "
                "not callable. Provide a function with the signature "
                "f(t, y, x_in, **params)."
            )

        reserved = tuple(self.exclude_kwargs_postprocessing)
        parameter_names = list(inspect.signature(func).parameters.keys())

        if all(arg in parameter_names for arg in reserved):
            # function follows the current signature
            self._check_reserved_parameters(
                func=func,
                reserved=reserved,
                n_leading=len(reserved),
                func_type="post-processing",
                expected_signature=f"f({', '.join(reserved)}, **params)",
            )
            return func

        return self.wrap_legacy_post_processing(func)

    def wrap_legacy_post_processing(self, func: Callable) -> Callable:
        """Wrap a legacy post-processing function ``f(results, time, interpolation,
        *params)`` so that it can be called as ``f(t, y, x_in, **params)``.

        The first three positional parameters of the legacy function are mapped
        by position (``results -> y``, ``time -> t``, ``interpolation -> x_in``).
        The wrapper accepts ``t``, ``y`` and ``x_in`` either as keywords or
        positionally (in the order ``t, y, x_in``). Any additional positional
        arguments and keyword arguments are forwarded to the legacy function
        after the first three arguments.
        """
        reserved = tuple(self.exclude_kwargs_postprocessing)
        missing = [a for a in LEGACY_POST_PROCESSING_MAP if a not in reserved]
        if len(missing) > 0:
            raise PymobError(
                f"Legacy post-processing function '{getattr(func, '__name__', func)}' "
                f"cannot be wrapped, because the arguments {missing} are not "
                f"listed in exclude_kwargs_postprocessing={reserved}. Legacy "
                "post-processing functions require the arguments "
                f"{LEGACY_POST_PROCESSING_MAP}."
            )

        signature = inspect.signature(func)
        parameters = list(signature.parameters.values())
        positional_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        n_legacy = len(LEGACY_POST_PROCESSING_MAP)
        legacy_positional = [
            p for p in parameters[:n_legacy] if p.kind in positional_kinds
        ]

        if len(legacy_positional) < n_legacy:
            raise PymobError(
                f"The post-processing function '{getattr(func, '__name__', func)}"
                f"{signature}' could not be validated. "
                "\n\n"
                "Why does this error occur?\n"
                "--------------------------\n"
                "Post-processing functions must have the signature "
                f"f({', '.join(reserved)}, **params). For backwards compatibility, "
                "legacy functions with the signature "
                "f(results, time, interpolation, *params) are also accepted, "
                "where the first three positional arguments are mapped by "
                "position (results -> y, time -> t, interpolation -> x_in). "
                f"The provided function neither contains all arguments {reserved} "
                "nor at least three leading positional arguments."
                "\n\n"
                "How to fix this error?\n"
                "----------------------\n"
                "Change the signature of the post-processing function to "
                f"f({', '.join(reserved)}, **params), e.g.: \n"
                f"`def post_processing({', '.join(reserved)}, my_param): ...`"
            )

        self._check_reserved_parameters(
            func=func,
            reserved=reserved,
            n_leading=n_legacy,
            func_type="legacy post-processing",
            expected_signature="f(results, time, interpolation, *params)",
        )

        remaining_parameters = parameters[n_legacy:]

        # construct the signature of the wrapper, so that `mappar` and other
        # inspection-based utilities see (t, y, x_in, *params)
        new_parameters = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in reserved
        ]
        for p in remaining_parameters:
            if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                p = p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            new_parameters.append(p)
        new_signature = signature.replace(parameters=new_parameters)

        @functools.wraps(func)
        def legacy_post_processing_wrapper(*args, **kwargs):
            args = list(args)
            # t, y and x_in may be given as keywords (then all positional args
            # are parameters) or positionally in the order (t, y, x_in)
            core = {name: kwargs.pop(name) for name in reserved if name in kwargs}
            for name in reserved:
                if name not in core:
                    if len(args) == 0:
                        raise TypeError(
                            f"Post-processing function '{func.__name__}' missing "
                            f"required argument: '{name}'"
                        )
                    core[name] = args.pop(0)

            legacy_args = [core[name] for name in LEGACY_POST_PROCESSING_MAP]
            return func(*legacy_args, *args, **kwargs)

        # __signature__ takes precedence over __wrapped__ in inspect.signature
        legacy_post_processing_wrapper.__signature__ = new_signature # type: ignore
        legacy_post_processing_wrapper.__pymob_legacy_post_processing__ = True # type: ignore

        return legacy_post_processing_wrapper


# Positional order of the legacy post-processing signature
# f(results, time, interpolation, *params) expressed in the names of the
# current signature f(t, y, x_in, **params): results -> y, time -> t,
# interpolation -> x_in
LEGACY_POST_PROCESSING_MAP: Tuple[str, str, str] = ("y", "t", "x_in")

# Translation of legacy argument names in exclude_kwargs_postprocessing
LEGACY_POST_PROCESSING_NAME_MAP: Dict[str, str] = {
    "results": "y",
    "time": "t",
    "interpolation": "x_in"
}


def mappar(
    func, 
    parameters: Dict[str,float|int|List|Tuple], 
    exclude=[], 
    to:Literal["tuple","dict","names"]="tuple"
) -> Tuple|Dict:
    func_signature = inspect.signature(func).parameters.keys()
    model_param_signature = [p for p in func_signature if p not in exclude]
    if to == "tuple":
        model_args = [parameters.get(k) for k in model_param_signature]
        return tuple(model_args)
    elif to == "dict":
        return {k: parameters.get(k) for k in model_param_signature}
    elif to == "names":
        return tuple(model_param_signature)

    raise NotImplementedError(f"'to={to}' is not implemented for 'mappar'")


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
    data_vars = tuple(x_in.data_vars.keys())
    dataset_dims = tuple(x_in.dims.keys())

    for k in data_vars:
        v_orig = x_in[k]
        v = v_orig.transpose(x_dim, ...)
        x = v[x_dim].values
        y = v.values
        x_dim_loc = v.dims.index(x_dim)

        ts_ = np.concatenate([x, np.array(x[-1]+1, ndmin=1)]) 
        if y.ndim == 1:
            ys_ = np.concatenate([y, np.array(y[-1], ndmin=1)])
        elif y.ndim == 2:
            if x_dim_loc == 0:
                ys_ = np.row_stack([y, np.expand_dims(y[-1], axis=0)]) 
        elif y.ndim == 3:
            if x_dim_loc == 0:
                ys_ = np.row_stack([y, np.expand_dims(y[-1], axis=0)])
            else:
                raise ValueError(f"{x_dim_loc} should be the first dimension.")
        else:
            raise NotImplementedError(
                "Dimensions of interpolation > 2 or 0 are not implemented"
            )

        xs, ys = rectilinear_interpolation(ts=ts_, ys=ys_) # type:ignore
        # if y.ndim <= 2:
        #     xs = xs
        # else:
        #     xs = xs[0] # type:ignore

        coords = {x_dim: xs}
        coords.update({d: v.coords[d].values for d in v.dims if d != x_dim})
        # coords = {d:coords[d] for d in v.dims}

        y_reindexed = xr.DataArray(
            ys, 
            coords=coords,
            name=v.name
        )
        y_reindexed = y_reindexed.transpose(*v_orig.dims)

        data_arrays.append(y_reindexed)

    x_in_interpolated = xr.combine_by_coords(data_objects=data_arrays)
    x_in_interpolated = x_in_interpolated.transpose(*dataset_dims)


    return x_in_interpolated[[*dataset_dims, *data_vars]]
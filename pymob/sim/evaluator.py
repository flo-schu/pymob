from typing import Callable, Dict, List, Optional, Sequence
import inspect
from frozendict import frozendict
from copy import deepcopy
import xarray as xr
import numpy as np
from pymob.solvers.base import mappar, SolverBase

def create_dataset_from_numpy(Y, Y_names, coordinates):
    DeprecationWarning(
        "This method will be discontinued in future relases. "
        "Use 'create_dataset_from_dict' instead and return a dictionary from "
        "the solver or post processing respectively. The needed variable names "
        "to create the dictionary can be obtained from the data_variables "
        "argument in the solver signature. "
    )
    n_vars = Y.shape[-1]
    n_dims = len(Y.shape)
    assert n_vars == len(Y_names), (
        "The number of datasets must be the same as the specified number"
        "of data variables declared in the `settings.cfg` file."
    )

    # transpose Y to put the variable dimension first, then add the
    # remaining dimensions in order
    Y_transposed = Y.transpose((n_dims - 1, *range(n_dims - 1)))

    data_arrays = []
    for y, y_name in zip(Y_transposed, Y_names):
        da = xr.DataArray(y, coords=coordinates, name=y_name)
        data_arrays.append(da)

    dataset = xr.merge(data_arrays)

    return dataset

def create_dataset_from_dict(Y: dict, data_structure, coordinates):
    arrays = {}
    for k, v in Y.items():
        dims = data_structure.get(k, tuple(coordinates.keys()))
        coords = {d: coordinates[d] for d in dims}
        da = xr.DataArray(v, coords=coords, dims=dims)
        arrays.update({k: da})

    return xr.Dataset(arrays)

class Evaluator:
    """The Evaluator is an instance to evaluate a model. It's purpose is primarily
    to create objects that can be spawned and evaluated in parallel and can 
    individually track the results of a simulation or a parameter inference
    process. If needed the evaluations can be tracked and results can later
    be collected.
    
    Seed may not be set as a property, because this should be something passed
    through
    """
    result: xr.Dataset

    def __init__(
            self,
            model: Callable,
            solver: type|Callable,
            dimensions: Sequence[str],
            n_ode_states: int,
            var_dim_mapper: Dict,
            data_structure: Dict,
            data_structure_and_dimensionality: Dict,
            coordinates: Dict,
            coordinates_input_vars: Dict,
            data_variables: Sequence[str],
            stochastic: bool,
            indices: Dict = {},
            post_processing: Optional[Callable] = None,
            **kwargs
        ) -> None:
        """_summary_

        Parameters
        ----------
        model : Callable
            the ODE model to be solved by the evaluator
        solver : Callable
            a function to solve the ODE model with
        parameters : Dict
            A dictionary of model and post_processing parameters. Do not have
            to be in any particular order
        dimensions : List
            A list of the dimensions of the simulations
        n_ode_states : int
            The number of ODE states tracked
        var_dim_mapper : List
            A list of variables and their associated dimensions. This is relevant
            for simulations, where not all data variables have the same dimensional
            structure
        data_structure : Dict
            Similar to the var_dim_mapper, but additionally contains the coordinates
        coordinates : Dict
            The coordinates of each dimension in a dict
        data_variables : List
            The data variables of the simulation
        stochastic : bool
            Whether the model is a stochastic or a deterministic model
        indices : Optional[Dict], optional
            Indices, which should be used to map potentially nested parameters
            to a flat array for batch processing the simulations, by default {}
        post_processing : Optional[Callable], optional
            A function that takes a dictionary of simulation results and 
            parameters as an input and adds new variables to the results, 
            by default None, meaning that no post processing of the ODE solution
            is performed
        """
        
        self._parameters = frozendict()
        self.dimensions = dimensions
        self.n_ode_states = n_ode_states
        self.var_dim_mapper = var_dim_mapper
        self.data_structure = data_structure
        self.data_structure_and_dimensionality = data_structure_and_dimensionality
        self.data_variables = data_variables
        self.coordinates = coordinates
        self.is_stochastic = stochastic
        self.indices = indices
        
        # can be initialized
        if post_processing is None:
            self.post_processing = lambda results, time, interpolation: results
        else: 
            self.post_processing = post_processing
                
        # can be initialized
        # set additional arguments of evaluator
        _ = [setattr(self, key, val) for key, val in kwargs.items()]

        self._signature = {}

        if callable(model):
            if hasattr(model, "__func__"):
                self.model = model.__func__
            else:
                self.model = model
        else:
            raise NotImplementedError(
                f"The model {model} must be provided as a callable."
            )

        # can be initialized
        if isinstance(solver, type):
            if issubclass(solver, SolverBase):
                frozen_coordinates_input_vars = frozendict({
                    k: frozendict({ck: tuple(cv) for ck, cv  in v.items()}) 
                    for k, v in coordinates_input_vars.items()
                })

                data_structure_dims = frozendict({
                    dv: frozendict({d: lendim for d, lendim in dimdict.items()}) 
                    for dv, dimdict 
                    in self.data_structure_and_dimensionality.items()
                })

                frozen_coordinates = frozendict({
                    k: tuple(v) for k, v in self.coordinates.items()
                })

                self._solver = solver(
                    model=self.model,
                    post_processing=self.post_processing,
                    solver_kwargs=frozendict({k:v for k, v in kwargs.items() if k in solver.extra_attributes}),
                    
                    coordinates=frozen_coordinates,
                    coordinates_input_vars=frozen_coordinates_input_vars,
                    dimensions=tuple(self.dimensions),
                    data_variables=tuple(self.data_variables),
                    data_structure_and_dimensionality=data_structure_dims,

                    indices=frozendict({k: tuple(v.values) for k, v in self.indices.items()}),
                    n_ode_states=self.n_ode_states,
                    is_stochastic=self.is_stochastic,
                )
            else:
                raise NotImplementedError(
                    f"If solver is passed as a class of type {type(solver)}. "
                    "Must be a subclass of `pymob.solvers.base.SolverBase`. "
                    "Alternatively pass a callable."
                )
        elif callable(solver):
            if hasattr(solver, "__func__"):
                self._solver = solver.__func__
            else:
                self._solver = solver
            self.get_call_signature()

        else:
            raise NotImplementedError(
                f"Solver {solver} is neither a subclass of "
                "`pymob.solvers.base.SolverBase` nor a callable."
            )

    # can be initialized
    def get_call_signature(self):
        if isinstance(self._solver, SolverBase):
            signature = inspect.signature(self._solver.solve)
        elif inspect.isfunction(self._solver) or inspect.ismethod(self._solver):
            signature = inspect.signature(self._solver)
        else:
            raise TypeError(f"{self._solver} must be SolverBase class or a function")
        
        model_args = [a for a in signature.parameters.keys() if a != "parameters"]

        for a in model_args:
            if a not in self.allowed_model_signature_arguments:
                raise ValueError(
                    f"'{a}' in model signature is not an attribute of the Evaluator. "
                    f"Use one of {self.allowed_model_signature_arguments}, "
                    f"or set as evaluator_kwargs in the call to "
                    "'SimulationBase.dispatch'" 
                )
            
            # add argument to signature for call to model
            if a != "seed":
                self._signature.update({a: getattr(self, a)})
        
    
    @property
    def allowed_model_signature_arguments(self):
        return [a for a in self.__dict__.keys() if a[0] != "_"] + ["seed"]

    def __call__(self, seed=None):
        if seed is not None:
            self._signature.update({"seed": seed})

        if isinstance(self._solver, SolverBase):
            Y_ = self._solver(**self.parameters)

        else:
            Y_ = self._solver(parameters=self.parameters, **self._signature)
        
        # TODO: Consider which elements may be abstracted from the solve methods 
        # implemented in mod.py below is an unsuccessful approach
        # params = self._signature["parameters"]["parameters"]
        # time = self._signature["coordinates"]["time"]
        
        # s_dim, s_idx = self._signature["indices"]["substance"]
        # pp_args = mappar(self.post_processing, params, exclude=["t", "results"])
        # pp_args = [np.array(a, ndmin=1)[s_idx] for a in pp_args]
        # Y_ = self.post_processing(Y_, time, *pp_args)
        self.Y = Y_

    @property
    def dimensionality(self):
        return {key: len(values) for key, values in self.coordinates.items()}

    @property
    def parameters(self) -> frozendict:
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: Dict):
        if len(self._parameters) == 0:
            self._parameters = frozendict(value)
        elif value == self._parameters:
            pass
        else:
            raise ValueError(
                "It is unsafe to change the parameters of an evaluator "
                "After it has been created. Use 'sim.dispatch(theta=...)' "
                "to create a new evaluator and initialize it with a new set "
                "of parameters."
                "If you really need to do it, use evaluator._parameters to "
                "overwrite the parameters on your own risk."
            )



    @property
    def results(self):
        if isinstance(self.Y, dict):
            return create_dataset_from_dict(
                Y=self.Y, 
                coordinates=self.coordinates,
                data_structure=self.data_structure,
            )
        elif isinstance(self.Y, np.ndarray):
            return create_dataset_from_numpy(
                Y=self.Y,
                Y_names=self.data_variables,
                coordinates=self.coordinates,
            )
        else:
            raise NotImplementedError(
                "Results returned by the solver must be of type Dict or np.ndarray."
            )
    
    def spawn(self):
        return deepcopy(self)
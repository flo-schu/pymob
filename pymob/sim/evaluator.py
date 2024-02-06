from typing import Callable, Dict, List, Optional
import inspect
import xarray as xr
import numpy as np

def create_dataset_from_numpy(Y, Y_names, coordinates):
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
            solver: Callable,
            parameters: Dict,
            dimensions: List,
            n_ode_states: int,
            var_dim_mapper: List,
            data_structure: Dict,
            coordinates: Dict,
            data_variables: List,
            stochastic: bool,
            indices: Optional[Dict] = {},
            post_processing: Optional[Callable] = None,
            **kwargs
        ) -> None:
        
        self.model = model
        self.parameters = parameters
        self.dimensions = dimensions
        self.n_ode_states = n_ode_states
        self.var_dim_mapper = var_dim_mapper
        self.data_structure = data_structure
        self.data_variables = data_variables
        self.coordinates = coordinates
        self.is_stochastic = stochastic
        self.indices = indices
        
        if post_processing is None:
            self.post_processing = lambda x: x
        else: 
            self.post_processing = post_processing
                
        # set additional arguments of evaluator
        _ = [setattr(self, key, val) for key, val in kwargs.items()]

        self._signature = {}

        if isinstance(solver, type):
            self._solver = solver()
        else:
            self._solver = solver

        self.get_call_signature()



    def get_call_signature(self):
        signature = inspect.signature(self._solver)
        model_args = list(signature.parameters.keys())

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

        Y_ = self._solver(**self._signature)
        Y_ = self.post_processing(Y_)
        self.Y = Y_

    @property
    def dimensionality(self):
        return {key: len(values) for key, values in self.coordinates.items()}

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
    
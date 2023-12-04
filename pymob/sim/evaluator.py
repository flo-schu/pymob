from typing import Callable, Dict, List, Optional, Union
import xarray as xr

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

class EvaluatorBase:
    result: xr.Dataset

    def __init__(
            self,
            model: Callable,
            y0: Optional[Union[List, xr.Dataset]],
            parameters: Dict,
            dimensions: List,
            coordinates: Dict,
            data_variables: List,
            seed: Optional[int] = None,
            **kwargs
        ) -> None:
        
        self.y0 = y0
        self.parameters = parameters
        self.dimensions = dimensions
        self.data_variables = data_variables
        self.coordinates = coordinates
        self.seed = seed
        
        if isinstance(model, type):
            self.model = model()
        else:
            self.model = model

    def __call__(self):
        self.Y = self.model(self)

    @property
    def results(self):
        return create_dataset_from_numpy(
            Y=self.Y, 
            Y_names=self.data_variables, 
            coordinates=self.coordinates
        )
    
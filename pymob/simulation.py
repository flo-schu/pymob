import os
import warnings
from typing import Optional, List, Union
import configparser
import multiprocessing as mp
from multiprocessing.pool import ThreadPool, Pool

import numpy as np
import xarray as xr
import dpath as dp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from toopy import Param, FloatParam, IntParam

from pymob.utils.errors import errormsg
from pymob.utils.store_file import scenario_file, parse_config_section, converters
from pymob.sim.config import Config

def update_parameters_dict(config, x, parnames):
    for par, val, in zip(parnames, x):
        key_exist = dp.set(config, glob=par, value=val, separator=".")
        if key_exist != 1:
            raise KeyError(
                f"prior parameter name: {par} was not found in config. " + 
                f"make sure parameter name was spelled correctly"
            )
    return config



class SimulationBase:
    def __init__(
        self, 
        config: Optional[Union[str,configparser.ConfigParser]] = None, 
    ) -> None:
        
        self.config = Config(config=config)
        self.model_parameters: tuple = ()
        self._observations: xr.Dataset = xr.Dataset()
        self._objective_names: Union[str, List[str]] = []
        self._coordinates: dict = {}
        self.free_model_parameters: list = []

        # draw some seeds
        self._seed_buffer_size: int = self.config.multiprocessing.n_cores * 2
        self.RNG = np.random.default_rng(self.config.simulation.seed)
        self._random_integers = self.create_random_integers(n=self._seed_buffer_size)
        
    def setup(self):
        """Simulation setup routine, when the following methods have been 
        defined:
        
        init-methods
        ------------

        self.initialize --> may be replaced by self.set_observations

        """

        self.initialize(input=self.config.input_file_paths)
        
        # coords = self.set_coordinates(input=self.config.input_file_paths)
        # self.coordinates = self.create_coordinates(coordinate_data=coords)
        self.free_model_parameters  = self.set_free_model_parameters()

        output_dir = self.config.case_study.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        scenario_dir = os.path.dirname(self.config.case_study.settings)
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir)
            print(f"Created directory: {scenario_dir}")

        # TODO: set up logger

    @property
    def observations(self):
        assert isinstance(self._observations, xr.Dataset), "Observations must be an xr.Dataset"
        return self._observations

    @observations.setter
    def observations(self, value):
        self._observations = value
        self.create_data_scaler()
        self.coordinates = self.set_coordinates(input=self.config.input_file_paths)
        

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = self.create_coordinates(coordinate_data=value)


    def __repr__(self) -> str:
        return (
            f"Simulation(case_study={self.config.case_study.name}, "
            f"scenario={self.config.case_study.scenario})"
        )


    def set_coordinates(self, input):
        dimensions = self.config.simulation.dimensions
        return [self.observations[dim].values for dim in dimensions]

    def evaluate(self, theta):
        """Wrapper around run to modify paramters of the model.
        """
        self.model_parameters = self.parameterize(self.config.input_file_paths, theta)
        return self.run()
    
    def compute(self):
        """
        A wrapper around run, which catches errors, logs, does post processing
        """
        warnings.warn("Discouraged to use self.Y constructs. Instability suspected.", DeprecationWarning, 2)
        self.Y = self.evaluate(theta=self.model_parameter_dict)

    def interactive(self):
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        def interactive_output(func, controls):
            out = widgets.Output(layout={'border': '1px solid black'})
            def observer(change):
                theta={key:s.value for key, s in sliders.items()}
                widgets.interaction.show_inline_matplotlib_plots()
                with out:
                    clear_output(wait=True)
                    func(theta)
                    widgets.interaction.show_inline_matplotlib_plots()
            for k, slider in controls.items():
                slider.observe(observer, "value")
            widgets.interaction.show_inline_matplotlib_plots()
            observer(None)
            return out

        sliders = {}
        for par in self.free_model_parameters:
            s = widgets.FloatSlider(
                par.value, description=par.name, min=par.min, max=par.max,
                step=par.step
            )
            sliders.update({par.name: s})

        def func(theta):
            self.Y = self.evaluate(theta)
            self.plot()

        out = interactive_output(func=func, controls=sliders)

        display(widgets.HBox([widgets.VBox([s for _, s in sliders.items()]), out]))
    
    def set_inferer(self, backend):
        if backend == "pyabc":
            from pymob.inference.pyabc_backend import PyabcBackend

            self.inferer = PyabcBackend(simulation=self)

        elif backend == "pymoo":
            from pymob.inference.pymoo_backend import PymooBackend

            self.inferer = PymooBackend(simulation=self)

        else:
            raise NotImplementedError("Inference backend is not implemented.")
        
    def dataset_to_2Darray(self, dataset: xr.Dataset) -> xr.DataArray: 
        array_2D = dataset.stack(multiindex=self.config.simulation.dimensions)
        return array_2D.to_array().transpose("multiindex", "variable")

    def array2D_to_dataset(self, dataarray: xr.DataArray) -> xr.Dataset: 
        dataset_2D = dataarray.to_dataset(dim="variable")      
        return dataset_2D.unstack().transpose(*self.config.simulation.dimensions)

    def create_data_scaler(self):
        """Creates a scaler for the data variables of the dataset over all
        remaining dimensions.
        In addition produces a scaled copy of the observations
        """
        # make sure the dataset follows the order of variables specified in
        # the config file. This is important so also in the simulation results
        # the scalers are matched.
        ordered_dataset = self.observations[self.config.simulation.data_variables]
        obs_2D_array = self.dataset_to_2Darray(dataset=ordered_dataset)
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        
        # add bounds to array of observations and fit scaler
        lower_bounds = np.array(self.config.simulation.data_variables_min)
        upper_bounds = np.array(self.config.simulation.data_variables_max)
        stacked_array = np.row_stack([lower_bounds, upper_bounds, obs_2D_array])
        scaler.fit(stacked_array)

        self.scaler = scaler
        self.print_scaling_info()

        scaled_obs = self.scale_(self.observations)
        self.observations_scaled = scaled_obs

    def print_scaling_info(self):
        scaler = type(self.scaler).__name__
        for i, var in enumerate(self.config.simulation.data_variables):
            print(
                f"{scaler}(variable={var}, "
                f"min={self.scaler.data_min_[i]}, max={self.scaler.data_max_[i]})"
            )

    def scale_(self, dataset: xr.Dataset):
        ordered_dataset = dataset[self.config.simulation.data_variables]
        data_2D_array = self.dataset_to_2Darray(dataset=ordered_dataset)
        obs_2D_array_scaled = data_2D_array.copy() 
        obs_2D_array_scaled.values = self.scaler.transform(data_2D_array) # type: ignore
        return self.array2D_to_dataset(obs_2D_array_scaled)

    @property
    def results(self):
        warnings.warn("Discouraged to use results property.", DeprecationWarning, 2)
        return self.create_dataset_from_numpy(
            Y=self.Y, 
            Y_names=self.config.simulation.data_variables, 
            coordinates=self.coordinates
        )

    def results_to_df(self, results):
        return self.create_dataset_from_numpy(
            Y=results, 
            Y_names=self.config.simulation.data_variables, 
            coordinates=self.coordinates
        )

    @property
    def results_scaled(self):
        scaled_results = self.scale_(self.results)
        # self.check_scaled_results_feasibility(scaled_results)
        return scaled_results

    def scale_results(self, Y):
        ds = self.create_dataset_from_numpy(
            Y=Y, 
            Y_names=self.config.simulation.data_variables, 
            coordinates=self.coordinates
        )
        return self.scale_(ds)

    def check_scaled_results_feasibility(self, scaled_results):
        """Parameter inference or optimization over many variables can only succeed
        in reasonable time if the results that should be compared are on approximately
        equal scales. The Simulation class, automatically estimates the scales
        of result variables, when observations are provided. 

        Problems can occurr when observations are on very narrow ranges, but the 
        simulation results can take much larger or lower values for that variable.
        As a result the inference procedure will almost exlusively focus on the
        optimization of this variable, because it provides the maximal return.

        The function warns the user, if simulation results largely deviate from 
        the scaled minima or maxima of the observations. In this case manual 
        minima and maxima should be given
        """
        max_scaled = scaled_results.max()
        min_scaled = scaled_results.min()
        if isinstance(self.scaler, MinMaxScaler):
            for varkey, varval in max_scaled.variables.items():
                if varval > 2:
                    warnings.warn(
                        f"Scaled results for '{varkey}' are {float(varval.values)} "
                        "above the ideal maximum of 1. "
                        "You should specify explicit bounds for the results variable."
                    )

            for varkey, varval in min_scaled.variables.items():
                if varval < -1:
                    warnings.warn(
                        f"Scaled results for '{varkey}' are {float(varval.values)} "
                        "below the ideal minimum of 0. "
                        "You should specify explicit bounds for the results variable."
                    )

    def validate(self):
        # TODO: run checks if the simulation was set up correctly
        #       - do observation dimensions match the model output (run a mini
        #         simulation with reduced coordinates to verify)
        #       -
        if len(self.config.simulation.data_variables) == 0:
            raise RuntimeError(
                "No data_variables were specified. "
                "Specify like sim.config.simulation.data_variables = ['a', 'b'] "
                "Or in the simulation section of the config file. "
                "Data variables track the state variables of the simulation. "
                "If you want to do inference, they must match the variables of "
                "the observations."
            )

                    
        if len(self.config.simulation.dimensions) == 0:
            raise RuntimeError(
                "No dimensions of the simulation were specified. "
                "Which observations are you expecting? "
                "'time' or 'id' are reasonable choices. But it all depends on "
                "your data. Dimensions must match your data if you want to do "
                "Parameter inference."
            )

    def parameterize(self, input: list[str], theta: list[Param]):
        """
        Optional. Set parameters of the model. By default returns an empty tuple. 
        Can be used to define parameters directly in the script or from a 
        parameter file.

        Arguments
        ---------

        input: List[str] file paths of parameter/input files
        theta: List[Param] a list of Parameters. By default the parameters
            specified in the settings.cfg are used in this list. 

        returns
        -------

        tulpe: tuple of parameters, can have any length.
        """
        return {p.name: p.value for p in theta}, 

    def run(self):
        """
        Implementation of the forward simulation of the model. Needs to return
        X and Y

        returns
        -------

        X: np.ndarray | xr.DataArray
        Y: np.ndarray | xr.DataArray
        """
        raise NotImplementedError
    
    def objective_function(self, results, **kwargs):
        func = getattr(self, self.config.inference.objective_function)
        obj = func(results, **kwargs)

        if obj.ndim == 0:
            obj_value = float(obj)
            obj_name = "objective"
        elif obj.ndim == 1:
            obj_value = obj.values
            obj_name = list(obj.coords["variable"].values)
        else:
            raise ValueError("Objectives should be at most 1-dimensional.")

        if len(self._objective_names) == 0:
            self._objective_names = obj_name

        return obj_name, obj_value

    def total_average(self, results):
        """objective function returning the total MSE of the entire dataset"""
        diff = (self.scale_results(results) - self.observations_scaled).to_array()
        return (diff ** 2).mean()

    def prior(self):
        raise NotImplementedError

    def initialize(self, input):
        """
        initializes the simulation. Performs any extra work, not done in 
        parameterize or set_coordinates. 
        """
        pass
    
    def dump(self):
        pass
        
    
    def plot(self):
        pass

    def create_coordinates(self, coordinate_data):
        if not isinstance(coordinate_data, (list, tuple)):
            coordinate_data = (coordinate_data, )

        assert len(self.config.simulation.dimensions) == len(coordinate_data), errormsg(
            f"""number of dimensions, specified in the configuration file
            must match the coordinate data (X) returned by the `run` method.
            """
        )

        coord_zipper = zip(self.config.simulation.dimensions, coordinate_data)
        coords = {dim: x_i for dim, x_i in coord_zipper}
        return coords

    @staticmethod
    def create_dataset_from_numpy(Y, Y_names, coordinates):
        n_vars = Y.shape[-1]
        n_dims = len(Y.shape)
        assert n_vars == len(Y_names), errormsg(
            """The number of datasets must be the same as the specified number
            of data variables declared in the `settings.cfg` file.
            """
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

    @property
    def model_parameter_values(self):
        return [p.value for p in self.free_model_parameters]
    
    @property
    def model_parameter_names(self):
        return [p.name for p in self.free_model_parameters]
    
    @property
    def n_free_parameters(self):
        return len(self.free_model_parameters)

    @property
    def model_parameter_dict(self):
        return {p.name:p.value for p in self.free_model_parameters}


    def create_random_integers(self, n: int):
        return self.RNG.integers(low=0, high=int(1e18), size=n).tolist()
        
    def refill_consumed_seeds(self):
        n_seeds_left = len(self._random_integers)
        if n_seeds_left == self.config.multiprocessing.n_cores:
            n_new_seeds = self._seed_buffer_size - n_seeds_left
            new_seeds = self.create_random_integers(n=n_new_seeds)
            self._random_integers.extend(new_seeds)
            print(f"Appended {n_new_seeds} new seeds to sim.")
        
    def draw_seed(self):
        return None       
        # the collowing has no multiprocessing stability
        # self.refill_consumed_seeds()
        # seed = self._random_integers.pop(0)
        # return seed

    def set_free_model_parameters(self) -> list:
        try:
            params = self.config.model_parameters.model_dump()
            
            # create a nested dictionary from model parameters
            parameter_dict = {}
            for par_key, par_value in params.items():
                dp.new(parameter_dict, par_key, par_value, separator=".")

            parse = lambda x: None if x is None else float(x)

            # create Param instances
            parameters = []
            for param_name, param_dict in parameter_dict.items():
                p = FloatParam(
                    value=parse(param_dict.get("value")),
                    name=param_name,
                    min=parse(param_dict.get("min")),
                    max=parse(param_dict.get("max")),
                    step=parse(param_dict.get("step")),
                    prior=param_dict.get("prior", None)
                )
                parameters.append(p)

            return parameters
        except KeyError:
            return []
        
            

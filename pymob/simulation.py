import os
import warnings
import configparser
import multiprocessing as mp
from typing import Callable
from multiprocessing.pool import ThreadPool, Pool

import numpy as np
import xarray as xr
import dpath.util as dp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from toopy import Param, FloatParam, IntParam

from pymob.utils.errors import errormsg
from pymob.utils.store_file import scenario_file, parse_config_section
from pymob.sim.evaluator import EvaluatorBase

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
    model: Callable
    def __init__(
        self, 
        config: configparser.ConfigParser, 
    ) -> None:
        
        self.config = config
        self.model_parameters: tuple = ()
        self.observations = None
        self._objective_names = []
        self._seed_buffer_size = self.n_cores * 2
        
        # seed gloabal RNG
        self.RNG = np.random.default_rng(self.seed)
        # draw a selection of 1e8 integers for using those throughout the
        # simulation
        self._random_integers = self.create_random_integers(n=self._seed_buffer_size)

        self.initialize(input=self.input_file_paths)
        
        if self.observations is not None:
            self.create_data_scaler()
        
        coords = self.set_coordinates(input=self.input_file_paths)
        self.coordinates = self.create_coordinates(coordinate_data=coords)
        self.free_model_parameters  = self.set_free_model_parameters()

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.validate()
        # TODO: set up logger


    def __repr__(self) -> str:
        return f"Simulation(case_study={self.case_study}, scenario={self.scenario})"

    def print_config(self):
        print("Simulation configuration", end="\n")
        print("========================")
        for section in self.config.sections():
            print("\n")
            print(f"[{section}]", end="\n")

            for opt, val in self.config.items(section):
                print(f"{opt} = {val}", end="\n")

        print("========================", end="\n")


    def dispatch(self, theta, **evaluator_kwargs):
        """Dispatch an evaluator, which will compute the model at parameters
        (theta). Evaluators are advantageous, because they are easier serialized
        than the whole simulation object. Comparison can then happen back in 
        the simulation.

        Theoretically, this could also be used to constrain coordinates etc, 
        before evaluating.  
        """
        model_parameters = self.parameterize(self.input_file_paths, theta)
        
        # TODO: make sure the evaluator has all arguments required for solving
        # model
        # TODO: Check if model is bound. If yes extract
        if hasattr(self.model, "__func__"):
            model = self.model.__func__
        else:
            model = self.model

        evaluator = EvaluatorBase(
            model=model,
            y0=model_parameters["y0"],
            parameters=model_parameters["parameters"],
            dimensions=self.dimensions,
            data_variables=self.data_variables,
            coordinates=self.coordinates,
            seed=self.draw_seed(),
            **evaluator_kwargs
        )

        return evaluator

    def evaluate(self, theta):
        """Wrapper around run to modify paramters of the model.
        """
        self.model_parameters = self.parameterize(self.input_file_paths, theta)
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

        if backend == "pymoo":
            from pymob.inference.pymoo_backend import PymooBackend

            self.inferer = PymooBackend(simulation=self)

    def check_dimensions(self, dataset):
        """Check if dataset dimensions match the specified dimensions.
        TODO: Name datasets for referencing them in errormessages
        """
        ds_dims = list(dataset.dims.keys())
        in_dims = [k in self.dimensions for k in ds_dims]
        assert all(in_dims), IndexError(
            "Not all dataset dimensions, were not found in specified dimensions. "
            f"Settings(dims={self.dimensions}) != dataset(dims={ds_dims})"
        )
        
    def dataset_to_2Darray(self, dataset: xr.Dataset): 
        self.check_dimensions(dataset=dataset)
        array_2D = dataset.stack(multiindex=self.dimensions)
        return array_2D.to_array().transpose("multiindex", "variable")

    def array2D_to_dataset(self, dataarray: xr.DataArray): 
        dataset_2D = dataarray.to_dataset(dim="variable")      
        return dataset_2D.unstack().transpose(*self.dimensions)

    def create_data_scaler(self):
        """Creates a scaler for the data variables of the dataset over all
        remaining dimensions.
        In addition produces a scaled copy of the observations
        """
        # make sure the dataset follows the order of variables specified in
        # the config file. This is important so also in the simulation results
        # the scalers are matched.
        ordered_dataset = self.observations[self.data_variables]
        obs_2D_array = self.dataset_to_2Darray(dataset=ordered_dataset)
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        
        # add bounds to array of observations and fit scaler
        lower_bounds, upper_bounds = self.data_variable_bounds
        stacked_array = np.row_stack([lower_bounds, upper_bounds, obs_2D_array])
        scaler.fit(stacked_array)

        self.scaler = scaler
        self.print_scaling_info()

        scaled_obs = self.scale_(self.observations)
        self.observations_scaled = scaled_obs

    def print_scaling_info(self):
        scaler = type(self.scaler).__name__
        for i, var in enumerate(self.data_variables):
            print(
                f"{scaler}(variable={var}, "
                f"min={self.scaler.data_min_[i]}, max={self.scaler.data_max_[i]})"
            )

    def scale_(self, dataset: xr.Dataset):
        ordered_dataset = dataset[self.data_variables]
        data_2D_array = self.dataset_to_2Darray(dataset=ordered_dataset)
        obs_2D_array_scaled = data_2D_array.copy() 
        obs_2D_array_scaled.values = self.scaler.transform(data_2D_array)
        return self.array2D_to_dataset(obs_2D_array_scaled)

    @property
    def results(self):
        warnings.warn("Discouraged to use results property.", DeprecationWarning, 2)
        return self.create_dataset_from_numpy(
            Y=self.Y, 
            Y_names=self.data_variables, 
            coordinates=self.coordinates
        )

    def results_to_df(self, results):
        return self.create_dataset_from_numpy(
            Y=results, 
            Y_names=self.data_variables, 
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
            Y_names=self.data_variables, 
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
        pass

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
        func = getattr(self, self.objective)
        obj = func(results, **kwargs)

        if obj.ndim == 0:
            obj_value = float(obj)
            obj_name = "objective"
        if obj.ndim == 1:
            obj_value = obj.values
            obj_name = list(obj.coords["variable"].values)

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
        if not isinstance(coordinate_data, tuple):
            coordinate_data = (coordinate_data, )

        assert len(self.dimensions) == len(coordinate_data), errormsg(
            f"""number of dimensions, specified in the configuration file
            must match the coordinate data (X) returned by the `run` method.
            """
        )

        coords = {dim: x_i for dim, x_i in zip(self.dimensions, coordinate_data)}
        return coords

    @staticmethod
    def create_dataset_from_numpy(Y, Y_names, coordinates):
        warnings.warn(
            "Use `create_dataset_from_numpy` defined in sim.evaluator",
            category=DeprecationWarning
        )
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

    @staticmethod
    def option_as_list(opt):
        if not isinstance(opt, (list, tuple)):
            opt_list = [opt]
        else:
            opt_list = opt

        return opt_list

    @property
    def input_file_paths(self):
        paths_input_files = []
        for file in self.input_files:
            fp = scenario_file(file, self.case_study, self.scenario)
            paths_input_files.append(fp)

        observation_files = self.config.getlist("case-study", "observations", fallback=[])
        if isinstance(observation_files, str):
            observation_files = [observation_files]

        for file in observation_files:
            if not os.path.isabs(file):
                fp = os.path.join(self.data_path, file)
            else:
                fp = file
            paths_input_files.append(fp)

        return paths_input_files

    # config as properties
    @property
    def dimensions(self):
        dims = self.config.getlist("simulation", "dimensions")
        return self.option_as_list(dims)

    @property
    def data_variables(self):
        data_vars = self.config.getlist("simulation", "data_variables")
        return self.option_as_list(data_vars)

    @property
    def input_files(self):
        input_files = self.config.getlist("simulation", "input_files", fallback=[])
        return self.option_as_list(input_files)
  
    @property
    def case_study_path(self):
        return self.config.get("case-study", "package")

    @property
    def root_path(self):
        return self.config.get("case-study", "root")

    @property
    def case_study(self):
        return self.config.get("case-study", "name")

    @property
    def scenario(self):
        return self.config.get("case-study", "scenario")

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


    @property
    def output_path(self):
        output_path_ = self.config.get("case-study", "output")
        if not os.path.isabs(output_path_):
            return os.path.join(
                os.path.relpath(output_path_),
                os.path.relpath(self.case_study_path), 
                "results",
                self.scenario,
            )
        else:
            return os.path.abspath(output_path_)

    @property
    def data_path(self):
        data_path_ = self.config.get("case-study", "data")
        if not os.path.isabs(data_path_):
            return os.path.join(
                self.case_study_path, 
                os.path.relpath(data_path_)
            )
        else:
            return os.path.abspath(data_path_)
       

    @property
    def data_variable_bounds(self):
        lower_bounds = self.config.getlistfloat(
            "simulation", "data_variables_min", fallback=None
        )
        upper_bounds = self.config.getlistfloat(
            "simulation", "data_variables_max", fallback=None
        )
        if lower_bounds is None:
            lower_bounds = [float("nan")] * 4

        if upper_bounds is None:
            upper_bounds = [float("nan")] * 4

        if not len(lower_bounds) == len(upper_bounds) == len(self.data_variables):
            raise ValueError(
                "If bounds are provided, the must be provided for all data "
                "variables. If a bound for a variable is unknown, write 'nan' "
                "in the config file at the position of the variable. "
                "\nE.g.:"
                "\ndata_variables = A B C"
                "\ndata_variables_max = 4 nan 2"
                "\ndata_variables_min = 0 0 nan"
            )
        
        return np.array(lower_bounds), np.array(upper_bounds)

    @property
    def objective(self):
        return self.config.get("inference", "objective_function", fallback="total_average")

    @property
    def n_objectives(self):
        return self.config.getint("inference", "n_objectives", fallback=1)

    @property
    def objective_names(self):
        return self._objective_names

    @property
    def n_threads(self):
        return self.config.getint("multiprocessing", "threads", fallback=4)
    
    @property
    def n_cores(self):
        cpu_avail = mp.cpu_count()
        cpu_set = self.config.getint("multiprocessing", "cores", fallback=1)
        if cpu_set <= 0:
            return cpu_avail + cpu_set
        else: 
            return cpu_set
    
    @n_cores.setter
    def n_cores(self, value):
        self.config.set("multiprocessing", "cores", str(value))

    def create_random_integers(self, n):
        return self.RNG.integers(0, 1e8, n).tolist()
        
    def refill_consumed_seeds(self):
        n_seeds_left = len(self._random_integers)
        if n_seeds_left == self.n_cores:
            n_new_seeds = self._seed_buffer_size - n_seeds_left
            new_seeds = self.create_random_integers(n=n_new_seeds)
            self._random_integers.extend(new_seeds)
            print(f"Appended {n_new_seeds} new seeds to sim.")
        
    def draw_seed(self):
        # return None       
        # the collowing has no multiprocessing stability when the simulation is
        # serialized directly
        self.refill_consumed_seeds()
        seed = self._random_integers.pop(0)
        return seed

    @property
    def seed(self):
        return int(self.config.get("simulation", "seed", fallback=1))

    def set_free_model_parameters(self):
        try:
            params = parse_config_section(self.config["model-parameters"], method="strfloat")
            
            # create a nested dictionary from model parameters
            parameter_dict = {}
            for par_key, par_value in params.items():
                dp.new(parameter_dict, par_key, par_value, separator=".")

            # create Param instances
            parameters = []
            for param_name, param_dict in parameter_dict.items():
                p = FloatParam(
                    value=param_dict.get("value"),
                    name=param_name,
                    min=param_dict.get("min", None),
                    max=param_dict.get("max", None),
                    step=param_dict.get("step", None),
                    prior=param_dict.get("prior", None)
                )
                parameters.append(p)

            return parameters
        except KeyError:
            return {}
        
            

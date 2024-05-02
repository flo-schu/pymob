import os
import inspect
import warnings
import configparser
from functools import partial
import multiprocessing as mp
from typing import Callable, Dict
from multiprocessing.pool import ThreadPool, Pool
import re

import numpy as np
import xarray as xr
import dpath.util as dp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from toopy import param, benchmark

from pymob.utils.config import lambdify_expression, lookup_args
from pymob.utils.errors import errormsg, import_optional_dependency
from pymob.utils.store_file import scenario_file, parse_config_section
from pymob.sim.evaluator import Evaluator, create_dataset_from_dict, create_dataset_from_numpy
from pymob.sim.base import stack_variables

def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def flatten_parameter_dict(model_parameter_dict, exclude_params=[]):
    """Takes a dictionary of key value pairs where the values may be
    floats or arrays. It flattens the arrays and adds indexes to the keys.
    In addition a function is returned that back-transforms the flattened
    parameters."""
    parameters = model_parameter_dict

    flat_params = {}
    empty_map = {}
    for par, value in parameters.items():
        if par in exclude_params:
            continue
        if is_iterable(value):
            empty_map.update({par: np.full_like(value, fill_value=np.nan)})
            for i, subvalue in enumerate(value):
                subpar = f"{par}___{i}"
                flat_params.update({subpar: subvalue})

        else:
            flat_params.update({par: value})

        if par not in empty_map:
            empty_map.update({par: np.nan})


    def reverse_mapper(parameters):
        param_dict = empty_map.copy()

        for subpar, value in parameters.items():
            subpar_list = subpar.split("___")

            if len(subpar_list) > 1:
                par, par_index = subpar_list
                param_dict[par][int(par_index)] = value
            elif len(subpar_list) == 1:
                par, = subpar_list
                param_dict[par] = value

        return param_dict
    
    return flat_params, reverse_mapper


def update_parameters_dict(config, x, parnames):
    for par, val, in zip(parnames, x):
        key_exist = dp.set(config, glob=par, value=val, separator=".")
        if key_exist != 1:
            raise KeyError(
                f"prior parameter name: {par} was not found in config. " + 
                f"make sure parameter name was spelled correctly"
            )
    return config

def get_return_arguments(func):
    ode_model_source = inspect.getsource(func)
    
    # extracts last return statement of source
    return_statement = ode_model_source.split("\n")[-2]

    # extract arguments returned by ode_func
    return_args = return_statement.split("return")[1]

    # strip whitespace and separate by comma
    return_args = return_args.replace(" ", "").split(",")

    return return_args

class SimulationBase:
    model: Callable
    def __init__(
        self, 
        config: configparser.ConfigParser, 
    ) -> None:
        
        self.config = config
        self.model_parameters: Dict = {}
        self.observations = None
        self._objective_names = []
        self._seed_buffer_size = self.n_cores * 2
        self.indices = {}

        # seed gloabal RNG
        self.RNG = np.random.default_rng(self.seed)
        # draw a selection of 1e8 integers for using those throughout the
        # simulation
        self._random_integers = self.create_random_integers(n=self._seed_buffer_size)

     
        self.initialize(input=self.input_file_paths)
        self.n_ode_states = self.infer_ode_states()
        
        if self.observations is not None:
            self.create_data_scaler()
        
        coords = self.set_coordinates(input=self.input_file_paths)
        self.coordinates = self.create_coordinates(coordinate_data=coords)
        self.var_dim_mapper = self.create_dim_index()
        self.free_model_parameters  = self.set_free_model_parameters()

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.validate()
        # TODO: set up logger
        self.parameterize = partial(self.parameterize, model_parameters=self.model_parameters)


    def __repr__(self) -> str:
        return f"Simulation(case_study={self.case_study}, scenario={self.scenario})"

    def load_functions(self):
        _model = self.config["simulation"].get("model", fallback=None)
        if _model is not None:
            self.model = getattr(self.mod, _model)

        _solver = self.config["simulation"].get("solver", fallback=None)
        if _solver is not None:
            self.solver = getattr(self.mod, _solver)

    def print_config(self):
        print("Simulation configuration", end="\n")
        print("========================")
        for section in self.config.sections():
            print("\n")
            print(f"[{section}]", end="\n")

            for opt, val in self.config.items(section):
                print(f"{opt} = {val}", end="\n")

        print("========================", end="\n")

    def benchmark(self, n=100, **kwargs):
        evaluator = self.dispatch(theta=self.model_parameter_dict, **kwargs)
        evaluator(seed=1) 

        @benchmark
        def run_bench():
            for i in range(n):
                evaluator(seed=np.random.randint(1e6))
        
        print(f"\nBenchmarking with {n} evaluations")
        print(f"=================================")
        run_bench()
        print(f"=================================\n")
        
    def infer_ode_states(self):
        if self.n_ode_states is None:
            try: 
                return_args = get_return_arguments(self.model)
                n_ode_states = len(return_args)
                warnings.warn(
                    "The number of ODE states was not specified in "
                    "the config file [simulation] > 'n_ode_states = <n>'. "
                    f"Extracted the return arguments {return_args} from the "
                    "source code. "
                    f"Setting 'n_ode_states={n_ode_states}."
                )
            except:
                warnings.warn(
                    "The number of ODE states was not specified in "
                    "the config file [simulation] > 'n_ode_states = <n>' "
                    "and could not be extracted from the return arguments."
                )
                n_ode_states = None
        else:
            n_ode_states = self.n_ode_states

        return n_ode_states
        
    def dispatch(self, theta, **evaluator_kwargs):
        """Dispatch an evaluator, which will compute the model at parameters
        (theta). Evaluators are advantageous, because they are easier serialized
        than the whole simulation object. Comparison can then happen back in 
        the simulation.

        Theoretically, this could also be used to constrain coordinates etc, 
        before evaluating.  
        """
        model_parameters = self.parameterize(theta)
        
        # TODO: make sure the evaluator has all arguments required for solving
        # model
        # TODO: Check if model is bound. If yes extract
        if hasattr(self.solver, "__func__"):
            solver = self.solver.__func__
        else:
            solver = self.solver

        if hasattr(self.model, "__func__"):
            model = self.model.__func__
        else:
            model = self.model
        
        if self.solver_post_processing is not None:
            # TODO: Handle similar to solver and model
            post_processing = getattr(self.mod, self.solver_post_processing)
        else:
            post_processing = None

        stochastic = self.config.get("simulation", "modeltype", fallback=False)
            
        evaluator = Evaluator(
            model=model,
            solver=solver,
            parameters=model_parameters,
            dimensions=self.dimensions,
            n_ode_states=self.n_ode_states,
            var_dim_mapper=self.var_dim_mapper,
            data_structure=self.data_structure,
            data_variables=self.data_variables,
            coordinates=self.coordinates,
            # TODO: pass the whole simulation settings section
            stochastic=True if stochastic == "stochastic" else False,
            indices=self.indices,
            post_processing=post_processing,
            **evaluator_kwargs
        )

        return evaluator

    def parse_input(self, data=None, input=None, drop_dims=["time"]):
        """Parses a config string e.g. y=Array([0]) or a=b to a numpy array 
        and looks up symbols in the elements of data, where data items are
        key:value pairs of a dictionary, xarray items or anything of this form

        The values are broadcasted along the remaining dimensions in the obser-
        vations that have not been dropped. Input refers to the argument in
        the config file. 

        This method is useful to prepare y0s from observations or to broadcast
        starting values along batch dimensions.
        """
        # parse dims and coords
        input_dims = {k:v for k, v in self.observations.dims.items() if k not in drop_dims}
        input_coords = {k:v for k, v in self.observations.coords.items() if k in input_dims}
        
        input_list = self.config["simulation"].getlist(input, fallback=[])
        if isinstance(input_list, str):
            input_list = [input_list]

        input_dataset = xr.Dataset()
        for input_expression in input_list:
            key, expr = input_expression.split("=")
            
            func, args = lambdify_expression(expr)

            kwargs = lookup_args(args, data)

            value = func(**kwargs)

            if not isinstance(value, xr.DataArray):
                value.shape != tuple(input_dims.values())
                value = np.broadcast_to(value, tuple(input_dims.values()))
                value = xr.DataArray(value, coords=input_coords)

            else:
                value = xr.DataArray(value.values, coords=input_coords)

            input_dataset[key] = value


        return input_dataset


    def reshape_observations(self, observations, reduce_dim):
        """This method reduces the dimensionality of the observations. 
        Compiling xarray datasets from multiple experiments with different 
        IDs and different endpoints, lead to blown up datasets where 
        all combinations (even though they were not tested) are filled with
        NaNs. Reducing such artificial dimensions by flattening the arrays
        is the aim of this method. 

        TODO: There should be tests, whether the method is applicable (this
        may be already caught with the assertion)

        TODO: The method should be generally applicable
        """

        raise NotImplementedError(
            "reshape_observations is an experimental method. "
            "Using this method may have unexpected results."
        )
        
        # currently the method is still based on the damage-proxy project
        substances = observations.attrs[reduce_dim]

        stacked_obs = stack_variables(
            ds=observations.copy(),
            variables=["cext_nom", "cext", "cint"],
            new_coordinates=substances,
            new_dim="substance",
            pattern=lambda var, coord: f"{var}_{coord}"
        ).transpose(*self.dimensions, reduce_dim)

        # reduce cext_obs
        cext_nom = stacked_obs["cext_nom"]
        assert np.all(
            (cext_nom == 0).sum(dim=reduce_dim) == len(substances) - 1
        ), "There are mixture treatments in the SingleSubstanceSim."


        # VECTORIZED INDEXING IS THE KEY
        # https://docs.xarray.dev/en/stable/user-guide/indexing.html#vectorized-indexing

        # Defining a function to reduce the "reduce_dim" dimension to length 1
        def index(array, axis=None, **kwargs):
            # Check that there is exactly one non-zero value in 'substance'
            non_zero_count = (array != 0).sum(axis=axis)
            
            if not (non_zero_count == 1).all():
                raise ValueError(f"Invalid '{reduce_dim}' dimension. It should have exactly one non-zero value.")
            
            return np.where(array != 0)[axis]


        # Applying the reduction function using groupby and reduce
        red_data = stacked_obs.cext_nom
        new_dims = [d for d in red_data.dims if d != reduce_dim]

        reduce_dim_idx = red_data\
            .groupby(*new_dims)\
            .reduce(index, dim=reduce_dim)\
            .rename(f"{reduce_dim}_index")
        
        if stacked_obs.dims[reduce_dim] == 1:
            reduce_dim_idx = reduce_dim_idx.squeeze()
        
        reduce_dim_id_mapping = stacked_obs[reduce_dim]\
            .isel({reduce_dim: reduce_dim_idx})\
            .drop(reduce_dim)\
            .rename(f"{reduce_dim}_id_mapping")
        
        reduce_dim_idx = reduce_dim_idx.assign_coords({
            f"{reduce_dim}": reduce_dim_id_mapping
        })

        # this works because XARRAY is amazing :)
        stacked_obs["cext_nom"] = stacked_obs["cext_nom"].sel({reduce_dim: reduce_dim_id_mapping})
        stacked_obs["cext"] = stacked_obs["cext"].sel({reduce_dim: reduce_dim_id_mapping})
        stacked_obs["cint"] = stacked_obs["cint"].sel({reduce_dim: reduce_dim_id_mapping})
        
        # drop old dimension and add dimension as inexed dimension
        # this is necessary, as the reduced dimension needs to disappear from
        # the coordinates.
        stacked_obs = stacked_obs.drop_dims(reduce_dim)
        stacked_obs = stacked_obs.assign_coords({
            f"{reduce_dim}": reduce_dim_id_mapping,
            f"{reduce_dim}_index": reduce_dim_idx,
        })
        
        indices = {
            reduce_dim: reduce_dim_idx
        }

        return stacked_obs, indices


    def evaluate(self, theta):
        """Wrapper around run to modify paramters of the model.
        """
        self.model_parameters = self.parameterize(theta)
        return self.run()
    
    def compute(self):
        """
        A wrapper around run, which catches errors, logs, does post processing
        """
        warnings.warn("Discouraged to use self.Y constructs. Instability suspected.", DeprecationWarning, 2)
        self.Y = self.evaluate(theta=self.model_parameter_dict)

    def interactive(self):
        # optional imports
        extra = "'interactive' dependencies can be installed with pip install pymob[interactive]"
        widgets = import_optional_dependency("ipywidgets", errors="raise", extra=extra)
        if widgets is not None:
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
            extra = self.config.getlist("inference", "extra_vars", fallback=[])
            extra = [extra] if isinstance(extra, str) else extra
            extra_vars = {v: self.observations[v] for v in extra}
            evaluator = self.dispatch(theta=theta, **extra_vars)
            evaluator()
            self.plot(results=evaluator.results)

        out = interactive_output(func=func, controls=sliders)

        display(widgets.HBox([widgets.VBox([s for _, s in sliders.items()]), out]))
    
    def set_inferer(self, backend):
        extra = (
            "set_inferer(backend='{0}') was not executed successfully, because "
            "'{0}' dependencies were not found. They can be installed with "
            "pip install pymob[{0}]. Alternatively:"
        )

        if backend == "pyabc":
            pyabc = import_optional_dependency(
                "pyabc", errors="raise", extra=extra.format("pyabc")
            )
            if pyabc is not None:
                from pymob.inference.pyabc_backend import PyabcBackend
            
            self.inferer = PyabcBackend(simulation=self)

        elif backend == "pymoo":
            pymoo = import_optional_dependency(
                "pymoo", errors="raise", extra=extra.format("pymoo2")
            )
            if pymoo is not None:
                from pymob.inference.pymoo_backend import PymooBackend

            self.inferer = PymooBackend(simulation=self)

        elif backend == "numpyro":
            numpyro = import_optional_dependency(
                "numpyro", errors="raise", extra=extra.format("numpyro")
            )
            if numpyro is not None:
                from pymob.inference.numpyro_backend import NumpyroBackend

            self.inferer = NumpyroBackend(simulation=self)
    
        else:
            raise NotImplementedError(f"Backend: {backend} is not implemented.")

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
        if isinstance(results, xr.Dataset):
            return results
        elif isinstance(results, dict):
            return create_dataset_from_dict(
                Y=results, 
                coordinates=self.coordinates,
                data_structure=self.data_structure,
            )
        elif isinstance(results, np.ndarray):
            return create_dataset_from_numpy(
                Y=results,
                Y_names=self.data_variables,
                coordinates=self.coordinates,
            )
        else:
            raise NotImplementedError(
                "Results returned by the solver must be of type Dict or np.ndarray."
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

    @staticmethod
    def parameterize(free_parameters: list[param.Param], model_parameters) -> dict:
        """
        Optional. Set parameters and initial values of the model. 
        Must return a dictionary with the keys 'y0' and 'parameters'
        
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
        parameters = model_parameters["parameters"]
        y0 = model_parameters["y0"]

        parameters.update({p.name: p.value for p in free_parameters})
        return {"y0": y0, "parameters": parameters} 

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
        
        diff = (self.scale_(self.results_to_df(results)) - self.observations_scaled).to_array()
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
    def n_ode_states(self):
        n_ode_states = self.config.getint(
            "simulation", "n_ode_states", fallback=None
        )

        return n_ode_states
    
    @n_ode_states.setter
    def n_ode_states(self, n_ode_state):
        self.config.set("simulation", "n_ode_states", f"{n_ode_state}")

    @property
    def solver_post_processing(self):
        return self.config["simulation"].get("solver_post_processing", fallback=None)

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
    def scenario_path(self):
        return os.path.join(
            self.root_path, 
            self.case_study_path, 
            "scenarios", 
            self.scenario
        )

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
        if self.config.has_section("model-parameters"):
            warnings.warn(
                "config section 'model-parameters' is deprecated, "
                "use 'free-model-parameters' and 'fixed-model-parameters'", 
                DeprecationWarning
            )
            params = parse_config_section(
                self.config["model-parameters"], method="strfloat"
            )
        elif self.config.has_section("free-model-parameters"):
            params = parse_config_section(
                self.config["free-model-parameters"], method="strfloat"
            )
        else:
            warnings.warn("No parameters were specified.")
            params = {}

        
        # create a nested dictionary from model parameters
        parameter_dict = {}
        for par_key, par_value in params.items():
            dp.new(parameter_dict, par_key, par_value, separator=".")

        # create Param instances
        parameters = []
        for param_name, param_dict in parameter_dict.items():
            value = param_dict.get("value", 1)
            if isinstance(value, (int, float)):
                p = param.FloatParam(
                    value=value,
                    name=param_name,
                    min=param_dict.get("min", None),
                    max=param_dict.get("max", None),
                    step=param_dict.get("step", None),
                    prior=param_dict.get("prior", None)
                )
            else:
                # check for array notation
                pattern = r"(\d+(\.\d+)?(\s+\d+(\.\d+)?)*|\s*)"
                if re.fullmatch(pattern, value):
                    value = np.array([float(v) for v in value.split(" ")])
                    p = param.ArrayParam(
                        value=value,
                        name=param_name,
                        min=param_dict.get("min", None),
                        max=param_dict.get("max", None),
                        step=param_dict.get("step", None),
                        prior=param_dict.get("prior", None)
                    )
                else:
                    raise NotImplementedError(
                        f"Parameter specification '{value}' cannot be parsed."
                    )
            parameters.append(p)

        return parameters

    @property
    def fixed_model_parameters(self):
        fixed_parameters = {}
        params = parse_config_section(self.config["fixed-model-parameters"])
        for k, v in params.items():
            vlist = v.split(" ")
            floatlist = [float(v) for v in vlist]
            if len(vlist) == 1:
                v_ = floatlist[0]

            else: 
                v_ = np.array(floatlist)

            fixed_parameters.update({k: v_})

        return fixed_parameters

    @property
    def error_model(self):
        em = parse_config_section(self.config["error-model"], method="strfloat")
        return em

    @property
    def evaluator_dim_order(self):
        return self.config.getlist("simulation", "evaluator_dim_order", fallback=None)

    def create_dim_index(self):
        # TODO: If a dimensionality config seciton is implemented this function
        # may become superflous
        sim_dims = self.dimensions
        evaluator_dims = self.evaluator_dim_order
        obs_ordered = self.observations.transpose(*sim_dims)

        var_dim_mapper = {}
        for var in self.data_variables:
            obs_var_dims = obs_ordered[var].dims
            var_dim_mapper.update({
                var: [obs_var_dims.index(e_i) for e_i in evaluator_dims if e_i in obs_var_dims]
            })

        return var_dim_mapper
    
    @property
    def data_structure(self):
        # TODO: If a dimensionality config seciton is implemented this function
        # may become superflous
        obs_ordered = self.observations.transpose(*self.dimensions)

        data_structure = {}
        for var in self.data_variables:
            obs_var_dims = obs_ordered[var].dims
            data_structure.update({
                var: list(obs_var_dims)
            })

        return data_structure

    def reorder_dims(self, Y):
        results = {}
        for var, mapper in self.var_dim_mapper.items():
            results.update({
                var: Y[var][np.array(mapper)]
            })
    
        return results

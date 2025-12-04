import os
import re
import sys
from ast import literal_eval as make_tuple
import configparser
import warnings
import importlib
import importlib.util
import multiprocessing as mp
from typing import List, Optional, Union, Dict, Literal, Callable, Tuple, TypedDict, Any
from typing_extensions import Annotated
from types import ModuleType
import tempfile
import click

import numpy as np
import xarray as xr

from pydantic import (
    BaseModel, Field, computed_field, model_validator, field_validator,
    ConfigDict, TypeAdapter, ValidationError
)
from pydantic.functional_validators import BeforeValidator
from pydantic.functional_serializers import PlainSerializer

from pymob.sim.casestudy_registry import get_case_study_model, _registry
from pymob.utils.store_file import scenario_file, converters
from pymob.sim.parameters import Param, NumericArray, OptionRV
# this loads at the import of the module
default_path = sys.path.copy()


class PymobModel(BaseModel):
    """
    Base class for all configuration models used in *pymob*.

    This class inherits from :class:`pydantic.BaseModel` and provides
    dictionary-style access to its fields.

    Parameters
    ----------
    **kwargs : dict
        Field values passed to the underlying :class:`BaseModel`.

    Examples
    --------
    >>> class MyModel(PymobModel):
    ...     a: int = 1
    ...     b: str = "x"
    >>> m = MyModel()
    >>> m["a"]
    1
    >>> m["b"] = "y"
    >>> m.b
    'y'
    """
    def __getitem__(self, key: str) -> Any:
        """Allow getting attribute by key (like a dict)."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow setting attribute by key (like a dict), with validation."""
        if key not in self.__annotations__:
            raise KeyError(f"Key '{key}' not found")

        # Set the value to the attribute
        setattr(self, key, value)

        # Trigger Pydantic validation, like when setting the attribute directly
        try:
            # Re-validate the model instance after setting the value
            self.__class__.parse_obj(self.dict())
        except ValidationError as e:
            raise ValueError(f"Validation failed: {e}")


class ModelParameterDict(TypedDict):
    """
    Typed dictionary used for model parameters.

    Attributes
    ----------
    parameters : dict
        Mapping of parameter names to values (float, str or int).
    y0 : xr.Dataset
        Initial conditions for the ODE system.
    x_in : xr.Dataset
        Input exposure time series.
    """
    parameters: Dict[str, float | str | int]
    y0: xr.Dataset
    x_in: xr.Dataset


class DataVariable(BaseModel):
    """
    Describe a data variable used in a case study.

    Parameters
    ----------
    dimensions : List[str]
        Dimensions that must be present in the observations and the
        dimensional order of the data variable.
    min : float, optional
        Minimum value used for scaling. ``np.nan`` means the minimum is taken
        from the observations. Default is ``np.nan``.
    max : float, optional
        Maximum value used for scaling. ``np.nan`` means the maximum is taken
        from the observations. Default is ``np.nan``.
    observed : bool, optional
        Whether the variable was observed. Default is ``True``.
    dimensions_evaluator : List[str] or None, optional
        Order of dimensions returned by the evaluator. If ``None`` the order
        of ``dimensions`` is used.

    Examples
    --------
    >>> dv = DataVariable(dimensions=["time", "species"])
    >>> dv.dimensions
    ['time', 'species']
    >>> dv.min = 0.0
    >>> dv.max = 1.0
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    dimensions: List[str]
    min: float = np.nan
    max: float = np.nan
    observed: bool = True
    dimensions_evaluator: Optional[List[str]] = None

    @model_validator(mode="after")
    def post_update(self) -> "DataVariable":
        """Validate that ``dimensions_evaluator`` matches ``dimensions``."""
        if self.dimensions_evaluator is not None:
            if len(self.dimensions_evaluator) != len(self.dimensions):
                self.dimensions_evaluator = None
        return self

    @field_validator("dimensions_evaluator", mode="after")
    def set_data_variable_bounds(cls, v, info, **kwargs) -> Optional[List[str]]:
        """
        Ensure that ``dimensions_evaluator`` has the same length and names as
        ``dimensions``.

        Raises
        ------
        AssertionError
            If the lengths differ or the sets of names differ.
        """
        dimensions = info.data.get("dimensions")
        if v is not None:
            if len(v) != len(dimensions):
                raise AssertionError(
                    f"Evaluator dimensions {v} must have the " +
                    f"same length as observations {dimensions} dimensions."
                )
            elif set(v) != set(dimensions):
                raise AssertionError(
                    f"Evaluator dimensions {set(v)} must have the " +
                    f"same names as observations {set(dimensions)} dimensions."
                )
            else:
                return v
        else:
            return None

    def __eq__(self, value: "DataVariable") -> bool:
        """Equality comparison that respects ``np.nan`` handling."""
        smin, smax = self.min, self.max
        vmin, vmax = value.min, value.max

        if np.all(np.isnan([smin, vmin])):
            min_equal = True
        else:
            min_equal = vmin == smin

        if np.all(np.isnan([smax, vmax])):
            max_equal = True
        else:
            max_equal = vmax == smax

        return all([
            self.dimensions == value.dimensions,
            self.observed == value.observed,
            self.dimensions_evaluator == value.dimensions_evaluator,
            min_equal,
            max_equal,
        ])


class ParameterDict(dict):
    """
    Dictionary that triggers a callback on mutation.

    This is used to keep the configuration in sync with the underlying
    simulation objects.

    Parameters
    ----------
    callback : Callable, optional
        Function called with the dictionary after each mutation.
    **kwargs : dict
        Initial key/value pairs.
    """
    def __init__(self, *args, **kwargs):
        self.callback: Callable = kwargs.pop('callback', None)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set ``key`` to ``value`` and invoke the callback."""
        super().__setitem__(key, value)
        if self.callback:
            self.callback(self)

    def update(self, *args, **kwargs) -> None:
        """Update the dictionary and invoke the callback."""
        super().update(*args, **kwargs)
        if self.callback:
            self.callback(self)


def string_to_list(option: Union[List, str]) -> List:
    """
    Convert a string or list/tuple to a list of strings.

    Parameters
    ----------
    option : Union[List, str]
        Input that may already be a list/tuple or a space-separated string.

    Returns
    -------
    List[str]
        List of strings.

    Examples
    --------
    >>> string_to_list("a b c")
    ['a', 'b', 'c']
    >>> string_to_list(["x", "y"])
    ['x', 'y']
    """
    if isinstance(option, (list, tuple)):
        return list(option)

    if len(option) == 0:
        return []
    elif " " not in option:
        return [option] 
    else:
        return [i.strip() for i in option.split(" ")]


def string_to_tuple(option: Union[List, str]) -> Tuple:
    """
    Convert a string or list/tuple to a tuple of strings.

    Parameters
    ----------
    option : Union[List, str]

    Returns
    -------
    Tuple[str, ...]
    """
    return tuple(string_to_list(option))


def string_to_dict(
        option: Union[Dict[str, str | float | int | List[float | int | str]], str]
    ) -> Dict[str, str | float | int | List[float | int | str]]:
    """
    Parse a configuration string into a dictionary.

    The string must consist of space-separated ``key=value`` pairs.
    Values are interpreted as ``float``, ``NumericArray`` or Python literals
    when possible.

    Parameters
    ----------
    option : Union[dict, str]
        Either an existing dictionary or a string to be parsed.

    Returns
    -------
    dict
        Mapping of keys to parsed values.

    Examples
    --------
    >>> string_to_dict("a=1 b=2")
    {'a': 1.0, 'b': 2.0}
    """
    if isinstance(option, Dict):
        return option

    retdict = {}
    if len(option) == 0:
        return retdict

    for i in option.split(" "):
        k, v = i.strip().split(sep="=", maxsplit=1)
        parsed = False
        if not parsed:
            try:
                parsed_value = TypeAdapter(float).validate_json(v)
                parsed = True
            except ValidationError:
                pass

        if not parsed:
            try:
                parsed_value = TypeAdapter(NumericArray).validate_json(v)
                parsed = True
            except ValueError:
                pass

        if not parsed and v[0] == "(" and v[-1] == ")":
            try:
                parsed_value = make_tuple(v)
                parsed = True
            except ValueError:
                pass

        # TODO: This expression seems to be wrong, but it causes no errors
        if not parsed and  v[0]=="[" and v[-1]=="]":
            try:
                v_ = v.strip("[]").split(",")
                # remove double quotes
                v_ = [re.sub(r'^"|"$', '', s) for s in v_]
                v_ = [re.sub(r"^'|'$", '', s) for s in v_]
                v_ = [v_i for v_i in v_ if v_i != ""]
                parsed_value = TypeAdapter(List[str]).validate_python(v_)
                parsed = True
            except ValidationError:
                pass

        if not parsed:
            parsed_value = v

        retdict.update({k: parsed_value})

    return retdict


def string_to_param(option: str | Param) -> Param:
    """
    Convert a string or :class:`Param` instance to a :class:`Param`.

    Parameters
    ----------
    option : str | Param

    Returns
    -------
    Param
    """
    if isinstance(option, Param):
        return option
    else:
        param_dict = string_to_dict(option)
        return Param.model_validate(param_dict, strict=False)


def string_to_datavar(option: str | DataVariable) -> DataVariable:
    """
    Convert a string or :class:`DataVariable` instance to a
    :class:`DataVariable`.

    Parameters
    ----------
    option : str | DataVariable

    Returns
    -------
    DataVariable
    """
    if isinstance(option, DataVariable):
        return option
    else:
        param_dict = string_to_dict(option)
        return DataVariable.model_validate(param_dict, strict=False)


def dict_to_string(dct: Dict, replace_whitespace="") -> str:
    """
    Serialize a dictionary to a space-separated ``key=value`` string.

    Parameters
    ----------
    dct : dict
        Dictionary to serialize.
    replace_whitespace : str, optional
        Replacement for whitespace inside values.

    Returns
    -------
    str
    """
    string_items = []
    for k, v in dct.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()

        expr = f"{k}={v}".replace(" ", replace_whitespace)
        string_items.append(expr)

    return " ".join(string_items)


def list_to_string(lst: List) -> str:
    """
    Serialize a list to a space-separated string.

    Parameters
    ----------
    lst : List

    Returns
    -------
    str
    """
    return " ".join([str(_l).replace(" ", "") for _l in lst])


def param_to_string(prm: Param) -> str:
    """
    Serialize a :class:`Param` to a ``key=value`` string.

    Parameters
    ----------
    prm : Param

    Returns
    -------
    str
    """
    return dict_to_string(prm.model_dump(exclude_none=True, mode="json"))


def datavar_to_string(prm: DataVariable) -> str:
    """
    Serialize a :class:`DataVariable` to a ``key=value`` string.

    Parameters
    ----------
    prm : DataVariable

    Returns
    -------
    str
    """
    return dict_to_string(prm.model_dump(exclude_none=True))


serialize_list_to_string = PlainSerializer(
    list_to_string, 
    return_type=str, 
    when_used="json"
)


serialize_dict_to_string = PlainSerializer(
    dict_to_string, 
    return_type=str, 
    when_used="json"
)


serialize_param_to_string = PlainSerializer(
    param_to_string, 
    return_type=str, 
    when_used="json"
)

serialize_datavar_to_string = PlainSerializer(
    datavar_to_string, 
    return_type=str, 
    when_used="json"
)

to_str = PlainSerializer(lambda x: str(x), return_type=str, when_used="json")


OptionListStr = Annotated[
    List[str], 
    BeforeValidator(string_to_list), 
    serialize_list_to_string
]

OptionTupleStr = Annotated[
    Tuple[str, ...], 
    BeforeValidator(string_to_tuple), 
    serialize_list_to_string
]

OptionDictStr = Annotated[
    Dict[str, str | float | int | List[float | int]], 
    BeforeValidator(string_to_dict), 
    serialize_dict_to_string
]

OptionDataVariable = Annotated[
    DataVariable, 
    BeforeValidator(string_to_datavar), 
    serialize_datavar_to_string
]

OptionParam = Annotated[
    Param, 
    BeforeValidator(string_to_param), 
    serialize_param_to_string
]


class Casestudy(PymobModel):
    """
    Configuration model for a case study.

    Attributes
    ----------
    init_root : str
        Working directory at the time of creation (excluded from serialization).
    root : str
        Root directory of the repository.
    name : str
        Name of the case study.
    version : Optional[str]
        Optional version string.
    pymob_version : Optional[str]
        Version of *pymob* used.
    scenario : str
        Scenario name.
    package : str
        Package containing the case study.
    modules : List[str]
        List of modules to import for the case study.
    simulation : str
        Name of the simulation class defined in ``sim.py``.
    output, data, scenario_path_override, observations : Optional[str]
        Paths data, scenario overrides and observation files.
    logging, logfile : str
        Logging level and optional logfile.
    """
    model_config = {"validate_assignment" : True}
    init_root: str = Field(default=os.getcwd(), exclude=True)
    root: str = "."
    name: str = "unnamed_case_study"
    version: Optional[str] = None
    pymob_version: Optional[str] = None
    scenario: str = "unnamed_scenario"
    package: str = "case_studies"
    modules: OptionListStr = ["sim", "mod", "prob", "data", "plot"]
    simulation: str = Field(default="Simulation", description="Simulation Class defined in sim.py module in the case study.")

    output: Optional[str] = None
    data: Optional[str] = None
    scenario_path_override: Optional[str] = None

    observations: Optional[str] = None
    
    logging: str = "DEBUG"
    logfile: Optional[str] = None

    @computed_field
    @property
    def output_path(self) -> str:
        """
        Path where results are stored.

        Returns
        -------
        str
        """
        if self.output is not None:
            return self.output
        else:
            return os.path.join(
                os.path.relpath(self.package),
                os.path.relpath(self.name),
                "results",
                self.scenario,
            )
    
    @output_path.setter
    def output_path(self, value: str) -> None:
        self.output = value

    @computed_field
    @property
    def data_path(self) -> str:
        """
        Path where case-study data files are stored.

        Returns
        -------
        str
        """
        if self.data is not None:
            return self.data
        else:
            return os.path.join(
                os.path.relpath(self.package), 
                os.path.relpath(self.name),
                "data"
            )
        
    @data_path.setter
    def data_path(self, value: str) -> None:
        self.data = value
    
    @computed_field
    @property
    def default_settings_path(self) -> str:
        """
        Default location of the ``settings.cfg`` file for the case study.

        Returns
        -------
        str
        """
        return os.path.join(
            os.path.relpath(self.package),
            os.path.relpath(self.name),
            "scenarios", 
            self.scenario, 
            "settings.cfg"
        )
        
    @property
    def scenario_path(self) -> str:
        """
        Path to the scenario directory, taking overrides into account.

        Returns
        -------
        str
        """
        if self.scenario_path_override is not None:
            return self.scenario_path_override

        package_path = os.path.basename(os.path.abspath(self.package))
        if package_path == self.name:
            # if the package path is identical to the case study path, then 
            # the scenario is directly located in the package.
            return os.path.join(
                os.path.relpath(self.package), 
                "scenarios", 
                self.scenario
            )
        else:
            return os.path.join(
                os.path.relpath(self.package), 
                os.path.relpath(self.name),
                "scenarios", 
                self.scenario
            )
    
    @field_validator("root", mode="after")
    def set_root(cls, new_value, info, **kwargs) -> str:
        """
        Validate and possibly change the working directory based on ``root``.

        Returns
        -------
        str
            The absolute path that becomes the new working directory.

        Note
        ----
        For conditionally updating values (e.g. when data variables change)
        see https://github.com/pydantic/pydantic/discussions/7127
        """
        os.chdir(info.data.get("init_root"))  # this resets the root to the original value
        package = info.data.get("package")
        name = info.data.get("name")
        root = os.path.abspath(new_value)
        if root != os.getcwd():
            if not os.path.exists(os.path.join(root, package)):
                raise FileNotFoundError(
                    f"Case study collection-directory '{package}' does not " +
                    f"exist in {root}. If the root is the case study. Set " +
                    "sim.config.package = '.'"
                )
            else:
                if not os.path.exists(os.path.join(root, package, name)):
                    raise FileNotFoundError(
                        f"Case study '{name}' " +
                        f"does not exist in {os.path.join(root, package)}. "
                    )   
                else:
                    os.chdir(root)
                    print(f"Working directory: '{root}'.")
                    return root
        else:
            print(f"Working directory: '{root}'.")
            return root


class Simulation(PymobModel):
    """
    Configuration model for a simulation.

    Attributes
    ----------
    model, solver : Optional[str]
        Names of the deterministic model and solver to use.
    y0, x_in, input_files : List[str]
        Lists of initial condition files, exposure files and other input files.
    n_ode_states : int
        Number of ODE states; ``-1`` means autodetect.
    batch_dimension, x_dimension : str
        Names of the batch and time dimensions.
    modeltype : Literal["stochastic", "deterministic"]
        Type of model.
    seed : int
        Random seed.
    """
    model_config = {"validate_assignment" : True, "extra": "allow"}

    model: Optional[str] = Field(default=None, validate_default=True, description="The deterministic model")
    solver: Optional[str] = Field(default=None, validate_default=True)
    
    y0: OptionListStr = []
    x_in: OptionListStr = []

    input_files: OptionListStr = []
    # data_variables: OptionListStr = []
    n_ode_states: int = -1
    batch_dimension: str = "batch_id"
    x_dimension: str = "time"
    modeltype: Literal["stochastic", "deterministic"] = "deterministic"
    solver_post_processing: Optional[str] = Field(default=None, validate_default=True)
    seed: Annotated[int, to_str] = 1


class Datastructure(PymobModel):
    """
    Container for data-variable definitions.

    The class stores ``DataVariable`` objects in ``__pydantic_extra__``.
    """
    __pydantic_extra__: Dict[str, OptionDataVariable]
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    def remove(self, key: str) -> None:
        """Removes a data variable from the data structure."""
        if key not in self.__pydantic_extra__:
            warnings.warn(
                f"'{key}' is not a data-variable. Data variables are: " +
                f"{self.data_variables}."
            )
            return
        
        deleted_var = self.__pydantic_extra__.pop(key)
        print(f"Deleted '{key}' DataVariable({deleted_var}).")

    def __getitem__(self, key: str) -> OptionDataVariable:
        return self.__pydantic_extra__[key]

    @property
    def data_variables(self) -> List[str]:
        """List of all data-variable names."""
        return [k for k in self.__pydantic_extra__.keys()]
    
    @property
    def observed_data_variables(self) -> List[str]:
        """List of observed data-variable names."""
        return [k for k, v in self.__pydantic_extra__.items() if v.observed]
    
    @property
    def dimensions(self) -> List[str]:
        """
        Legacy method returning the union of all dimensions.

        Returns
        -------
        List[str]

        Note
        ----
        TODO: Remove when dimensions is not accessed any longer
        """
        warnings.warn(
            "Legacy method, will be deprecated soon. This works only if all " +
            "Data variables have the same dimension",
            category=DeprecationWarning
        )
        dims = []
        for k, v in self.__pydantic_extra__.items():
            for d in v.dimensions:
                if d not in dims:
                    dims.append(d)
        return dims
    
    @property
    def all(self) -> Dict[str, DataVariable]:
        """Dictionary of all data variables."""
        return {k: v for k, v in self.__pydantic_extra__.items()}

    @property
    def dimdict(self) -> Dict[str, List[str]]:
        """Mapping of variable name to its dimensions."""
        return {k: v.dimensions for k, v in self.__pydantic_extra__.items()}

    @property
    def observed_dimdict(self) -> Dict[str, List[str]]:
        """Mapping of observed variable name to its dimensions."""
        return {k: v.dimensions for k, v in self.__pydantic_extra__.items() if v.observed}


    @property
    def var_dim_mapper(self) -> Dict[str, List[str]]:
        """
        Mapping from variable name to evaluator dimension indices.

        Returns
        -------
        Dict[str, List[int]]
        """
        var_dim_mapper = {}
        for k, v in self.all.items():
            if v.dimensions_evaluator is None:
                dims_evaluator = v.dimensions
            else:
                dims_evaluator = v.dimensions_evaluator

            var_dim_mapper.update({
                k: [v.dimensions.index(e_i) for e_i in dims_evaluator]
            })

        return var_dim_mapper
    
    @property
    def evaluator_dim_order(self) -> List[str]:
        """
        Legacy method returning the order of dimensions used by the evaluator.

        Returns
        -------
        List[str]

        Note
        ----
        TODO: Remove when dimensions is not accessed any longer
        """
        warnings.warn(
            "Legacy method, will be deprecated soon. This works only if all " +
            "Data variables have the same dimension",
            category=DeprecationWarning
        )
        dims = []
        for k, v in self.all.items():
            if v.dimensions_evaluator is None:
                dims_evaluator = v.dimensions
            else:
                dims_evaluator = v.dimensions_evaluator

            for d in dims_evaluator:
                if d not in dims:
                    dims.append(d)
        return dims
    
    @property
    def data_variables_min(self) -> List[float]:
        """List of ``min`` values for all data variables."""
        return [v.min for v in self.__pydantic_extra__.values()]

    @property
    def data_variables_max(self) -> List[float]:
        """List of ``max`` values for all data variables."""
        return [v.max for v in self.__pydantic_extra__.values()]

    @property
    def observed_data_variables_min(self) -> List[float]:
        """List of ``min`` values for observed data variables."""
        return [v.min for v in self.__pydantic_extra__.values() if v.observed]

    @property
    def observed_data_variables_max(self) -> List[float]:
        """List of ``max`` values for observed data variables."""
        return [v.max for v in self.__pydantic_extra__.values() if v.observed]


class Solverbase(PymobModel):
    """
    Base configuration for solvers.

    Attributes
    ----------
    x_dim : str
        Name of the time dimension.
    exclude_kwargs_model, exclude_kwargs_postprocessing : Tuple[str, ...]
        Keyword arguments to exclude when passing to the model or post-processing.
    """
    model_config = ConfigDict(
        validate_assignment=True, 
        extra="forbid"
    )
    x_dim: str = "time"
    exclude_kwargs_model: OptionTupleStr = ("t", "time", "x_in", "y", "x", "Y", "X")
    exclude_kwargs_postprocessing: OptionTupleStr = ("t", "time", "interpolation", "results")


class Jaxsolver(PymobModel):
    """
    Configuration for the JAX solver.

    Attributes
    ----------
    diffrax_solver : str
        Name of the Diffrax solver to use.
    rtol, atol : float
        Relative and absolute tolerances.
    pcoeff, icoeff, dcoeff, max_steps : float or int
        Coefficients for the solver.
    throw_exception : bool
        Whether to raise an exception on solver failure.
    """
    diffrax_solver: str = "Dopri5"
    rtol: float = 1e-6
    atol: float = 1e-7
    pcoeff: float = 0.0
    icoeff: float = 1.0
    dcoeff: float = 0.0
    max_steps: int = int(1e5)
    throw_exception: bool = True


class Inference(PymobModel):
    """
    Configuration for inference settings.

    Attributes
    ----------
    eps : float
        Numerical epsilon for convergence.
    objective_function : str
        Name of the objective function.
    n_objectives : int
        Number of objectives.
    objective_names : List[str]
        Names of the objectives.
    backend : Optional[str]
        Inference backend to use.
    extra_vars : List[str]
        Additional variables to monitor.
    plot : Optional[Callable]
        Plotting function.
    n_predictions : int
        Number of posterior predictive draws.
    """
    model_config = {"validate_assignment" : True}

    eps: float = 1e-8
    objective_function: str = "total_average"
    n_objectives: Annotated[int, to_str] = 1
    objective_names: OptionListStr = []
    backend: Optional[str] = None
    extra_vars: OptionListStr = []
    plot: Optional[str | Callable] = None
    n_predictions: Annotated[int, to_str] = 100


class Multiprocessing(PymobModel):
    """
    Configuration for multiprocessing.

    Attributes
    ----------
    cores : int
        Number of cores to use. ``1`` means no parallelism.
    """
    _name = "multiprocessing"
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # TODO: Use as private field
    cores: Annotated[int, to_str] = 1
    
    @property
    def n_cores(self) -> Annotated[int, to_str]:
        """
        Compute the effective number of cores.

        Returns
        -------
        int
        """
        cpu_avail = mp.cpu_count()
        cpu_set = self.cores
        if cpu_set <= 0:
            return cpu_avail + cpu_set
        else: 
            return cpu_set


class Modelparameters(PymobModel):
    """
    Container for model parameters.

    Parameters are stored in ``__pydantic_extra__`` as :class:`Param` objects.
    """
    __pydantic_extra__: Dict[str, OptionParam]
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @property
    def all(self) -> Dict[str, OptionParam]:
        """All parameters."""
        return self.__pydantic_extra__
    
    @all.setter
    def all(self, value: Dict[str, OptionParam]) -> None:
        self.__pydantic_extra__ = value

    @property
    def free(self) -> Dict[str, OptionParam]:
        """Parameters marked as free."""
        return {k: v for k, v in self.all.items() if v.free}

    @property
    def fixed(self) -> Dict[str, OptionParam]:
        """Parameters marked as fixed."""
        return {k: v for k, v in self.all.items() if not v.free}

    @property
    def n_free(self) -> int:
        """Number of free parameters."""
        return len(self.free)
    
    @property
    def free_value_dict(self) -> Dict[str, float | NumericArray]:
        """Dictionary of free parameter values."""
        return {k: v.value for k, v in self.free.items()}
    
    @property
    def fixed_value_dict(self) -> Dict[str, float | NumericArray]:
        """Dictionary of fixed parameter values."""
        return {k: v.value for k, v in self.fixed.items()}
    
    @property
    def value_dict(self) -> Dict[str, float | NumericArray]:
        """Dictionary of all parameter values."""
        return {k: v.value for k, v in self.all.items()}
    
    @property
    def dimensions(self) -> List[str]:
        """
        Legacy method returning the union of all parameter dimensions.

        Returns
        -------
        List[str]

        Note
        ----
        TODO: Remove when dimensions is not accessed any longer
        """
        warnings.warn(
            "Legacy method, will be deprecated soon. This works only if all " +
            "Data variables have the same dimension",
            category=DeprecationWarning
        )
        dims = []
        for k, v in self.all.items():
            for d in v.dims:
                if d not in dims:
                    dims.append(d)
        return dims

    def remove(self, key: str) -> None:
        """Removes a Parameter."""
        if key not in self.all:
            warnings.warn(
                f"'{key}' is not a parameter. Parameters are: " +
                f"{list(self.all.keys())}."
            )
            return
        
        deleted_par = self.all.pop(key)
        print(f"Deleted '{key}' Param({deleted_par}).")

    def reorder(self, keys: List[str]) -> None:
        """
        Reorders model parameters. This may be necessary for hierarchical 
        models, because priors take draws from hyperpriors to parameterize their
        distributions. Hence, they must be available earlier.

        Parameters
        ----------
        keys : List[str]
            Desired order of parameter keys. Unlisted keys are appended.
        """
        if len(keys) < len(self.all):
            keys = keys + [k for k in self.all.keys() if k not in keys]
            
        self.all = {k: self.all[k] for k in keys} 


class Errormodel(PymobModel):
    """
    Container for error-model specifications.

    Each entry is an :class:`OptionRV` describing a random variable.
    """
    __pydantic_extra__: Dict[str, OptionRV]
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    
    @property
    def all(self) -> Dict[str, OptionRV]:
        """All error-model random variables."""
        return self.__pydantic_extra__


class Pyabc(PymobModel):
    """
    Configuration for the pyABC inference backend.

    Attributes
    ----------
    sampler : str
        Sampler class name.
    population_size : int
        Number of particles per population.
    minimum_epsilon, min_eps_diff : float
        Epsilon settings.
    max_nr_populations : int
        Maximum number of populations.
    database_path : str
        Path to the SQLite database used by pyABC.
    """
    model_config = {"validate_assignment" : True}

    sampler: str = "SingleCoreSampler"
    population_size: Annotated[int, to_str] = 100
    minimum_epsilon: Annotated[float, to_str] = 0.0
    min_eps_diff: Annotated[float, to_str] = 0.0
    max_nr_populations: Annotated[int, to_str] = 1000
    
    # database configuration
    database_path: str = f"{tempfile.gettempdir()}/pyabc.db"


class Redis(PymobModel):
    """
    Configuration for the Redis backend used by pyABC.

    Attributes
    ----------
    password : str
        Redis password.
    port : int
        Port number.
    n_predictions, history_id, model_id : int
        Evaluation settings.
    """
    model_config = {"validate_assignment" : True, "protected_namespaces": ()}

    _name = "inference.pyabc.redis"

    # redis configuration
    password: str = "nopassword"
    port: Annotated[int, to_str] = 1111

    # eval configuration
    n_predictions: Annotated[int, to_str] = Field(default=50, alias="eval.n_predictions")
    history_id: Annotated[int, to_str] = Field(default=-1, alias="eval.history_id")
    model_id: Annotated[int, to_str] = Field(default=0, alias="eval.model_id")


class Pymoo(PymobModel):
    """
    Configuration for the pymoo multi-objective optimisation backend.

    Attributes
    ----------
    algortihm : str
        Optimisation algorithm.
    population_size, max_nr_populations : int
        Population settings.
    ftol, xtol, cvtol : float
        Tolerances.
    verbose : bool
        Verbosity flag.
    """
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    algortihm: str = "UNSGA3"
    population_size: Annotated[int, to_str] = 100
    max_nr_populations: Annotated[int, to_str] = 1000
    ftol: Annotated[float, to_str] = 1e-5
    xtol: Annotated[float, to_str] = 1e-7
    cvtol: Annotated[float, to_str] = 1e-7
    verbose: Annotated[bool, to_str] = True
    
class Numpyro(PymobModel):
    """
    Configuration for the NumPyro inference backend.

    Attributes
    ----------
    user_defined_probability_model, user_defined_error_model,
    user_defined_preprocessing : Optional[str]
        Paths to user-defined modules.
    gaussian_base_distribution : bool
        Whether to use a Gaussian base distribution.
    kernel, init_strategy : str
        MCMC kernel and initialisation strategy.
    chains, draws, warmup, thinning : int
        MCMC sampling parameters.
    nuts_draws, nuts_step_size, nuts_max_tree_depth,
    nuts_target_accept_prob, nuts_dense_mass,
    nuts_adapt_step_size, nuts_adapt_mass_matrix : bool or float
        NUTS sampler settings.
    sa_adapt_state_size : Optional        Simulated annealing state size.
    svi_iterations, svi_learning_rate : int, float
        SVI settings.
    """
    model_config = ConfigDict(validate_assignment=True, extra="ignore")
    user_defined_probability_model: Optional[str] = None
    user_defined_error_model: Optional[str] = None
    user_defined_preprocessing: Optional[str] = None
    gaussian_base_distribution: bool = False
    
    # inference arguments
    kernel: str = "nuts"
    init_strategy: str = "init_to_uniform"
     
    # mcmc parameters
    chains: Annotated[int, to_str] = 1
    draws: Annotated[int, to_str] = 2000
    warmup: Annotated[int, to_str] = 1000
    thinning: Annotated[int, to_str] = 1
    
    # nuts arguments
    nuts_draws: Annotated[int, to_str] = 2000
    nuts_step_size: Annotated[float, to_str] = 0.8
    nuts_max_tree_depth: int = 10
    nuts_target_accept_prob: Annotated[float, to_str] = 0.8
    nuts_dense_mass: Annotated[bool, to_str] = True
    nuts_adapt_step_size: Annotated[bool, to_str] = True
    nuts_adapt_mass_matrix: Annotated[bool, to_str] = True

    # sa parameters
    sa_adapt_state_size: Optional[int] = None

    # parameters
    svi_iterations: Annotated[int, to_str] = 10_000
    svi_learning_rate: Annotated[float, to_str] = 0.0001


class Report(PymobModel):
    """
    Configuration for report generation.

    Attributes
    ----------
    debug_report : bool
        Include debug information.
    pandoc_output_format : Literal["html", "latex-si", "latex", "pdf"]
        Output format for the generated report.
    model, parameters, diagnostics, goodness_of_fit,
    table_parameter_estimates, plot_trace, plot_parameter_pairs : bool
        Flags controlling which sections are included.
    """
    model_config = ConfigDict(validate_assignment=True, extra="ignore")
    
    debug_report: Annotated[bool, to_str] = False
    pandoc_output_format: Literal["html", "latex-si", "latex", "pdf"] = "html"

    model: Annotated[bool, to_str] = True
    parameters: Annotated[bool, to_str] = True
    parameters_format: Literal["xarray", "pandas"] = "pandas"
    
    diagnostics: Annotated[bool, to_str] = True
    diagnostics_with_batch_dim_vars: Annotated[bool, to_str] = False
    diagnostics_exclude_vars: OptionListStr = []
    
    goodness_of_fit: Annotated[bool, to_str] = True
    goodness_of_fit_use_predictions: Annotated[bool, to_str] = True
    goodness_of_fit_nrmse_mode: Literal["mean", "range", "iqr"] = "range"

    table_parameter_estimates: Annotated[bool, to_str] = True
    table_parameter_estimates_format: Literal["latex", "csv", "tsv"] = "csv"
    table_parameter_estimates_significant_figures: int = 3
    table_parameter_estimates_error_metric: Literal["hdi", "sd"] = "sd"
    table_parameter_estimates_parameters_as_rows: Annotated[bool, to_str] = True
    table_parameter_estimates_with_batch_dim_vars: Annotated[bool, to_str] = False
    table_parameter_estimates_exclude_vars: OptionListStr = []
    table_parameter_estimates_override_names: OptionDictStr = {}

    plot_trace: Annotated[bool, to_str] = True
    plot_parameter_pairs: Annotated[bool, to_str] = True


class Config(BaseModel):
    """
    Configuration manager for *pymob*.

    This class loads a ``settings.cfg`` file, validates it against the
    pydantic models defined above and provides convenient accessors.

    Parameters
    ----------
    config : str or configparser.ConfigParser, optional
        Path to a configuration file or an already parsed ``ConfigParser``.
    """
    model_config = {"validate_assignment" : True, "extra": "allow", "protected_namespaces": ()}
    _config: configparser.ConfigParser
    _modules: Dict[str, ModuleType]

    def __init__(
        self,
        config: Optional[Union[str, configparser.ConfigParser]] = None,
    ) -> None:

        _cfg_fp = None
        interp = configparser.ExtendedInterpolation()
        if isinstance(config, str):
            _config = configparser.ConfigParser(
                converters=converters,
                interpolation=interp
            )        
            _config.optionxform = str # type: ignore
            _cfg_file_paths = _config.read(config)
            try:
                _cfg_fp = _cfg_file_paths[0]
            except IndexError:
                raise FileNotFoundError(
                    f"Config file: {config} could not be found."
                ) 
        elif isinstance(config, configparser.ConfigParser):
            _config = config
        else:
            _config = configparser.ConfigParser(
                converters=converters,
                interpolation=interp
            )
            _config.optionxform = str # type: ignore

        # pass arguments to config

        if _cfg_fp is not None: 
            _config.set("case-study", "settings_path", _cfg_fp)
        cfg_dict = {k:dict(s) for k, s in dict(_config).items() if k != "DEFAULT"}
        
        # initalize case_study separately and import modules. This is done here,
        # so that the configuration sections from the registry are discoverable
        # this can only happen once the respective case studies are imported and with them
        # their settings models are defined registered (this of course must be done with the
        # register_case_study_config utility.
        _case_study_raw = cfg_dict.get("case-study", {})
        _case_study_instance = Casestudy.model_validate(_case_study_raw)
        _modules = self._import_casestudy_modules(case_study=_case_study_instance, reset_path=True)

        # initialize submodels
        super().__init__(**cfg_dict)

        self._config = _config
        self._modules = _modules

        # Load any case-study-specific configuration after generic sections are parsed
        self._load_case_study_section()

    case_study: Casestudy = Field(default=Casestudy(), alias="case-study")
    simulation: Simulation = Field(default=Simulation())
    data_structure: Datastructure = Field(default=Datastructure(), alias="data-structure") # type:ignore
    solverbase: Solverbase = Field(default=Solverbase())
    jaxsolver: Jaxsolver = Field(default=Jaxsolver(), alias="jax-solver")
    inference: Inference = Field(default=Inference())
    model_parameters: Modelparameters = Field(default=Modelparameters(), alias="model-parameters") #type: ignore
    error_model: Errormodel = Field(default=Errormodel(), alias="error-model") # type: ignore
    multiprocessing: Multiprocessing = Field(default=Multiprocessing())
    inference_pyabc: Pyabc = Field(default=Pyabc(), alias="inference.pyabc")
    inference_pyabc_redis: Redis = Field(default=Redis(), alias="inference.pyabc.redis")
    inference_pymoo: Pymoo = Field(default=Pymoo(), alias="inference.pymoo")
    inference_numpyro: Numpyro = Field(default=Numpyro(), alias="inference.numpyro")
    report: Report = Field(default=Report(), alias="report")
        
    @property
    def input_file_paths(self) -> List[str]:
        """
        List of all input files required for the simulation.

        Returns
        -------
        List[str]
        """
        paths_input_files = []
        for file in self.simulation.input_files:
            fp = scenario_file(file, self.case_study.name, self.case_study.scenario, pkg_dir=self.case_study.package)
            paths_input_files.append(fp)


        file = self.case_study.observations
        if file is None:
            return paths_input_files
        else:
            if not os.path.isabs(file):
                fp = os.path.join(self.case_study.data_path, file)
            else:
                fp = file
            paths_input_files.append(fp)

            return paths_input_files

    def print(self) -> None:
        """Print a summary of the configuration."""
        print("Simulation configuration", end="\n")
        print("========================")
        for section, field_info in self.model_fields.items():
            print(f"{section}({getattr(self, section)})", end="\n") # type: ignore

        print("========================", end="\n")

    def save(self, fp: Optional[str]=None, force: bool=False) -> None:
        """
        Save the configuration to a ``settings.cfg`` file.

        Uses serializers defined at the top, which parse the options to str
        so they can be processed by configfile. 

        In case the model configuration should be stored to a json file use
        something like `json.dumps(self.model_dump())`, because the build in
        function, is somewhat disabled by the listparsers which are needed for
        configfile lists.

        Parameters
        ----------
        fp : Optional[str]
            File path to write the settings file to. If ``None`` the default
            location derived from the case study is used.
        force : bool, optional
            Overwrite without prompting. Default is ``False``.
        """
        settings = self.model_dump(
            by_alias=True, 
            mode="json", 
            exclude_none=True,
            exclude={"case_study": {"output_path", "data_path", "root", "init_root", "default_settings_path"}}
        )
        self._config.update(**settings)

        if fp is None:
            file_path = self.case_study.default_settings_path
        else:
            file_path = os.path.abspath(fp)

        write = True
        if os.path.exists(file_path) and not force:
            ui_overwrite = input("Settings file already exists. Overwrite? [y/N]")
            write = True if ui_overwrite == "y" else False
        else:
            # create a directory for the new scenario file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if write:
            with open(file_path, "w") as f:
                self._config.write(f)

    def create_directory(self, directory: Literal["results", "scenario"], force: bool=False) -> None:
        """
        Create a results or scenario directory if it does not exist.

        Parameters
        ----------
        directory : Literal["results", "scenario"]
            Which directory to create.
        force : bool, optional
            If ``True`` create without prompting.
        """
        if directory == "results":
            p = os.path.abspath(self.case_study.output_path)
        elif directory == "scenario":
            p = os.path.abspath(self.case_study.scenario_path)
        else:
            raise NotImplementedError(
                f"{directory.capitalize()} is not an expected directory in the " +
                "case study logic. Use one of 'results' and 'scenario'."
            )

        if os.path.exists(p):
            print(f"{directory.capitalize()} directory exists at '{p}'.")
            return
        else:
            if not force:
                answer = input(f"Create {directory} directory at '{p}'? [Y/n]")
            else:
                answer = "y"

            if answer.lower() == "y" or answer.lower() == "":
                os.makedirs(p)
                print(f"{directory.capitalize()} directory created at '{p}'.")
            else:
                print(f"No {directory} directory created.")

    def import_casestudy_modules(self, reset_path: bool=False) -> None:
        """
        Import all modules of the current case study.

        this script handles the import of a case study without the typical 
        __init__.py file. It iterates over all .py files in the root directory
        of the case study (typically: sim, mod, stats, plot, data, prior)
        and imports them with import_module(...)

        Parameters
        ----------
        reset_path : bool, optional
            Reset ``sys.path`` before importing. Default is ``False``.

        """
        modules = self._import_casestudy_modules(
            case_study=self.case_study, 
            reset_path=reset_path
        )
        
        self._modules.update(modules)

    @staticmethod
    def _import_casestudy_modules(case_study: Casestudy, reset_path: bool=False) -> Dict[str, ModuleType]:
        """
        Import modules for a given case study.

        Parameters
        ----------
        case_study : Casestudy
            The case-study configuration.
        reset_path : bool, optional
            Whether to reset ``sys.path`` before importing.

        Returns
        -------
        Dict[str, ModuleType]
            Mapping of module name to imported module.
        """
        _modules = {}


        # reset the path to avoid importing modules form case-studies used
        # before in the same session
        if reset_path:
            # default path needs be copied, otherwise it will be updated
            # when setting sys.path
            sys.path = default_path.copy()


        # potential BUG: This is not safe. It is not guaranteed that the 
        # case study has the same name as the package. But it might be in the future
        package = case_study.name

        if "-" in package or " " in package:
            warnings.warn(
                f"Case-study contained {package} contained unallowed "+
                "characters: ['-', ' ']. "+
                "The characters will be replaced with underscores ('_') for "+
                "importing the package modules. " +
                "In the future, the name of the case study should be the same as " +
                "as the package where the modules are located. This name must not " +
                "contain hyphens ('-') or whitespace characters (' '),",
                category=UserWarning
            )
            _package = package.replace("-", "_").replace(" ", "_")
        else:
            _package = package

        spec = importlib.util.find_spec(_package)
        if spec is not None:
            for module in case_study.modules:
                try:
                    # TODO: Consider importing modules as a nested dictionary 
                    # with the indexing key being the package. The package
                    # cannot be derived from the class, if a method, that is 
                    # executed on a lower level case-study, should target that 
                    # a module belonging to the same package, because if the
                    # object is used, it would resolve to the package of the
                    # higher level case-study
                    m = importlib.import_module(f"{_package}.{module}")
                    _modules.update({module: m})
                except ModuleNotFoundError:
                    warnings.warn(
                        f"Module {module}.py not found in {_package}." +
                        "Missing modules can lead to unexpected behavior. " +
                        f"Does your case study have a {module}.py file? " +
                        "It should have the line `from PARENT_CASE_STUDY." +
                        f"{module} import *` to import all objects from " +
                        "the parent case study."
                    )
            return _modules

        # append relevant paths to sys
        package = os.path.join(
            case_study.root, 
            case_study.package
        )
        if package not in sys.path:
            sys.path.insert(0, package)
            print(f"Inserted '{package}' in PATH at index=0")
    
        case_study_path = os.path.join(
            case_study.root, 
            case_study.package,
            case_study.name,
            # Account for package architecture 
            case_study.name
        )
        if case_study_path not in sys.path:
            sys.path.insert(0, case_study_path)
            print(f"Inserted '{case_study_path}' in PATH at index=0")

        for module in case_study.modules:
            # remove modules of a different case study that might have been
            # loaded in the same session.
            if module in sys.modules:
                _ = sys.modules.pop(module)

        for module in case_study.modules:
            try:
                m = importlib.import_module(module, package=case_study_path)
                _modules.update({module: m})
            except ModuleNotFoundError:
                warnings.warn(
                    f"Module {module}.py not found in {case_study_path}." +
                    "Missing modules can lead to unexpected behavior." +
                    "If a module is not imported, you can specify it in the " +
                    "Config 'config.case_study.modules = [...]'"
                )

        return _modules

    def import_simulation_from_case_study(self) -> Any:
        """
        Retrieve the ``Simulation`` class defined in the case-study.

        Returns
        -------
        Any
            The ``Simulation`` class object.

        Raises
        ------
        ImportError
            If the class cannot be found.
        """
        try:
            Simulation = getattr(self._modules["sim"], self.case_study.simulation)
        except Exception as e:
            raise ImportError(
                f"Simulation class '{self.case_study.simulation}' " +
                "could not be found. Make sure the simulaton option is spelled " +
                "correctly or specify a class that exists in sim.py" +
                "If you are using pymob to work on different case-studies in " +
                "the same session, make sure to reset the path by " +
                "using `import_casestudy_modules(reset_path=True)`" +
                f"\nOriginal exception: {e}"
            )
        
        return Simulation

    def set_option(self, section: str, option: str, value: str) -> None:
        """
        Set a configuration option.

        Parameters
        ----------
        section : str
            Name of the configuration section (e.g. ``simulation``).
        option : str
            Option name within the section.
        value : str
            New value as a string; will be parsed according to the section's
            type definitions.
        """
        sect = getattr(self, section)
        if isinstance(sect, Modelparameters):
            if (value == "" or value == "None") and option in sect.all:
                sect.remove(option)
            elif (value == "" or value == "None") and option not in sect.all:
                pass
            else:
                sect[option] = value
        else:
            sect[option] = value

    def _load_case_study_section(self) -> None:
        """
        Parse a registered case-study-specific INI section and expose it as an attribute.
        """
        # Ensure the model has extra sections
        if self.model_extra is None:
            return

        # parse the config sections if they are defined in the settings.cfg
        # this overwrites the default definitions
        for section, options in self.model_extra.items():
            # get the model from the registry
            model_cls = get_case_study_model(section)

            if model_cls is None:
                warnings.warn(
                    f"Config section {section} could not be parsed. " +
                    "You need to register a the Config Model like that:\n\n" +
                    ">>> from pymob.sim.casestudy_registry import register_case_study_config\n" +
                    ">>> from pymob.sim.config import PymobModel\n" +
                    f">>> class {section.capitalize()}Config(PymobModel):\n" +
                    ">>>     option_a: float = 1.0\n" +
                    f">>> register_case_study_config({section.capitalize()}Config)\n"
                )

            else:
                # Pydantic parses/validates the raw strings
                instance = model_cls.model_validate(options, strict=False)
                # Store on the Config object – attribute name equals the case-study name
                self.model_extra.update({section: instance})

        # add default sections in case the sections are not defined in the settings.cfg
        for section, model_cls in _registry.items():
            if section not in self.model_extra:
                default_instance = model_cls.model_validate({}, strict=False)
                self.model_extra.update({section: default_instance})
            

@click.command
@click.option("--file", "-f", type=str, nargs=1, help="Path to the config file (usually in scenario/.../settings.cfg)")
@click.option("--options", "-o", type=str, multiple=True, help="The option or options to configure. Combine sections and option like this 'simulation.seed=1' options that have spaces need to be wraped in quotes")
def configure(file: str, options: Tuple[str, ...]) -> None:
    """
    Command-line entry point to modify a configuration file.

    Parameters
    ----------
    file : str
        Path to the configuration file.
    options : Tuple[str, ...]
        List of ``section.option=value`` strings.

    Examples
    --------
    >>> # Change the seed of the simulation
    >>> configure("-f", "scenario/settings.cfg", "-o", "simulation.seed=42")
    """
    config = Config(file)
    for opt in options:
        key, val = opt.split("=", 1)

        section, option = key.strip(" ").rsplit(".", 1)

        section = section.replace("-","_").replace(".","_")

        if section == "jax_solver":
            section = "jaxsolver"

        value = val.strip(" ")
        config.set_option(section, option, value)

    config.save(file, force=True)

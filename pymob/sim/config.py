import os
import re
import sys
from dataclasses import dataclass
from glob import glob
import configparser
import warnings
import importlib
import logging
import json
import multiprocessing as mp
from typing import List, Optional, Union, Dict, Literal, Callable
from typing_extensions import Annotated
from types import ModuleType
import tempfile

import numpy as np
from numpy.typing import ArrayLike

from pydantic import (
    BaseModel, Field, computed_field, field_validator, model_validator, 
    ConfigDict, TypeAdapter, ValidationError
)
from pydantic.functional_validators import BeforeValidator, AfterValidator
from pydantic.functional_serializers import PlainSerializer

from pymob.utils.store_file import scenario_file, converters


class FloatParam(BaseModel):
    name: Optional[str] = None
    value: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    prior: Optional[str] = None
    free: bool = True


class ArrayParam(BaseModel):
    name: Optional[str] = None
    value: List[float] = [0.0]
    min: Optional[List[float]] = None
    max: Optional[List[float]] = None
    step: Optional[List[float]] = None
    prior: Optional[str] = None
    free: bool = True

class DataVariable(BaseModel):
    """Describe a data variable

    Parameters
    ----------
    dimensions: List[str]
        Specifies the dimensions, which have to be found in the observations and 
        the dimensional order of the data variable
    
    min: float
        The possible minimum of a data variable, is used to construct scalers.
        Defaults to 'nan', which sets the minimum to the minimum of the 
        observations

    min: float
        The possible maxmimum of a data variable, is used to construct scalers
        Defaults to 'nan', which sets the maximum to the maximum of the 
        observations

    dimensions_evaluator: List[str]
        Specifies the dimensions and their order returned by the evaluator.
        This is necessary to bring observations and results together, if for some
        reason the returned simulation results from the evaluator have a different
        dimensional order than the observations

    Returns
    -------
    None

    """
    dimensions: List[str]
    min: float = np.nan
    max: float = np.nan
    dimensions_evaluator: Optional[List[str]] = None

    @model_validator(mode="after")
    def post_update(self):
        if self.dimensions_evaluator is not None:
            if len(self.dimensions_evaluator) != len(self.dimensions):
                self.dimensions_evaluator = None

        return self
        
        
    @field_validator("dimensions_evaluator", mode="after")
    def set_data_variable_bounds(cls, v, info, **kwargs):
        # For conditionally updating values (e.g. when data variables change)
        # see https://github.com/pydantic/pydantic/discussions/7127
        dimensions = info.data.get("dimensions")
        if v is not None:
            if len(v) != len(dimensions):
                raise AssertionError(
                    f"Evaluator dimensions {v} must have the "
                    f"same length as observations {dimensions} dimensions."
                )

            elif set(v) != set(dimensions):
                raise AssertionError(
                    f"Evaluator dimensions {set(v)} must have the "
                    f"same names as observations {set(dimensions)} dimensions."
                )

            else:
                return v
        else:
            return None
    


class ParameterDict(dict):
    def __init__(self, *args, **kwargs):
        self.callback: Callable = kwargs.pop('callback', None)
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.callback:
            self.callback(self)
    
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        if self.callback:
            self.callback(self)

def string_to_list(option: Union[List, str]) -> List:
    if isinstance(option, (list, tuple)):
        return list(option)
    
    if len(option) == 0:
        return []
    elif " " not in option:
        return [option] 
    else:
        return [i.strip() for i in option.split(" ")]


def string_to_dict(
        option: Union[Dict[str,str|float|int|List[float|int|str]], str]
    ) -> Dict[str,str|float|int|List[float|int|str]]:
    """Expects a string of this form 
    e.g. 'value=1 max=10 min=0 prior=Lognormal(loc=2,scale=1)'
    """
    if isinstance(option, Dict):
        return option
    
    else:
        retdict = {}
        for i in option.split(" "):
            k, v = i.strip().split(sep="=", maxsplit=1)
            parsed = False
            if not parsed:
                try:
                    parsed_value = TypeAdapter(List[float]).validate_json(v)
                    parsed = True
                except ValidationError:
                    pass

            if not parsed:
                try:
                    parsed_value = TypeAdapter(float).validate_json(v)
                    parsed = True
                except ValidationError:
                    pass

            if not parsed and  "[" in v and "]" in v:
                try:
                    v_ = v.strip("[]").split(",")
                    # remove double quotes
                    v_ = [re.sub(r'^"|"$', '', s) for s in v_]
                    v_ = [re.sub(r"^'|'$", '', s) for s in v_]
                    parsed_value = TypeAdapter(List[str]).validate_python(v_)
                    parsed = True
                except ValidationError:
                    pass

            if not parsed:
                parsed_value = v

            retdict.update({k:parsed_value})
                
        return retdict


def string_to_param(option:str|FloatParam|ArrayParam) -> FloatParam|ArrayParam:
    if isinstance(option, FloatParam|ArrayParam):
        return option
    else:
        param_dict = string_to_dict(option)
        if isinstance(param_dict.get("value"), float):
            return FloatParam.model_validate(param_dict, strict=False)
        
        else:
            return ArrayParam.model_validate(param_dict, strict=False)
    

def string_to_datavar(option:str|DataVariable) -> DataVariable:
    if isinstance(option, DataVariable):
        return option
    else:
        param_dict = string_to_dict(option)
        return DataVariable.model_validate(param_dict, strict=False)
        
    

def dict_to_string(dct: Dict):
    return " ".join([f"{k}={v}".replace(" ", "") for k, v in dct.items()]).replace("'", "")


def list_to_string(lst: List):
    return " ".join([str(l) for l in lst])


def param_to_string(prm: ArrayParam|FloatParam):
    return dict_to_string(prm.model_dump(exclude_none=True))

def datavar_to_string(prm: DataVariable):
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


OptionDictStr = Annotated[
    Dict[str,str|float|int|List[float|int]], 
    BeforeValidator(string_to_dict), 
    serialize_dict_to_string
]

OptionDataVariable = Annotated[
    DataVariable, 
    BeforeValidator(string_to_datavar), 
    serialize_datavar_to_string
]

OptionParam = Annotated[
    FloatParam|ArrayParam, 
    BeforeValidator(string_to_param), 
    serialize_param_to_string
]


OptionListFloat = Annotated[
    List[float], 
    BeforeValidator(string_to_list), 
    serialize_list_to_string
]

        
class Casestudy(BaseModel):
    model_config = {"validate_assignment" : True}
    init_root: str = Field(default=os.getcwd(), exclude=True)
    root: str = "."

    name: str = "unnamed_case_study"
    scenario: str = "unnamed_scenario"
    package: str = "case_studies"
    modules: OptionListStr = ["sim", "mod", "prob", "data", "plot"]
    simulation: str = Field(default="Simulation", description="Simulation Class defined in sim.py module in the case study.")

    output: Optional[str] = None
    data: Optional[str] = None

    observations: OptionListStr = []
    
    logging: str = "DEBUG"
    logfile: Optional[str] = None

    @computed_field
    @property
    def output_path(self) -> str:
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
    def output_path(self, value) -> None:
        self.output = value

    @computed_field
    @property
    def data_path(self) -> str:
        if self.data is not None:
            return self.data
        else:
            return os.path.join(
                os.path.relpath(self.package), 
                os.path.relpath(self.name),
                "data"
            )
        
    @data_path.setter
    def data_path(self, value) -> None:
        self.data = value
    
    @computed_field
    @property
    def default_settings_path(self) -> str:
        return os.path.join(
            os.path.relpath(self.package),
            os.path.relpath(self.name),
            "scenarios", 
            self.scenario, 
            "settings.cfg"
        )
        
    @property
    def scenario_path(self):
        return os.path.join(
            os.path.relpath(self.package), 
            os.path.relpath(self.name),
            "scenarios", 
            self.scenario
        )
    
    @field_validator("root", mode="after")
    def set_root(cls, new_value, info, **kwargs):
        # For conditionally updating values (e.g. when data variables change)
        # see https://github.com/pydantic/pydantic/discussions/7127
        os.chdir(info.data.get("init_root"))  # this resets the root to the original value
        package = info.data.get("package")
        name = info.data.get("name")
        root = os.path.abspath(new_value)
        if root != os.getcwd():
            if not os.path.exists(os.path.join(root, package)):
                raise FileNotFoundError(
                    f"Case study collection-directory '{package}' does not "
                    f"exist in {root}. If the root is the case study. Set "
                    f"sim.config.package = '.'"
                )
            else:
                if not os.path.exists(os.path.join(root, package, name)):
                    raise FileNotFoundError(
                        f"Case study '{name}' "
                        f"does not exist in {os.path.join(root, package)}. "
                    )   
                else:
                    os.chdir(root)
                    print(f"Working directory: '{root}'.")
                    return root
        else:
            print(f"Working directory: '{root}'.")
            return root




class Simulation(BaseModel):
    model_config = {"validate_assignment" : True, "extra": "allow"}

    model: Optional[str] = Field(default=None, validate_default=True, description="The deterministic model")
    solver: Optional[str] = Field(default=None, validate_default=True)
    
    y0: OptionListStr = []
    x_in: OptionListStr = []

    input_files: OptionListStr = []
    # data_variables: OptionListStr = []
    n_ode_states: int = -1
    replicated: bool = False
    modeltype: Literal["stochastic", "deterministic"] = "stochastic"
    solver_post_processing: Optional[str] = Field(default=None, validate_default=True)
    seed: Annotated[int, to_str] = 1

class Datastructure(BaseModel):
    __pydantic_extra__: Dict[str,OptionDataVariable]
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @property
    def data_variables(self) -> List[str]:
        return [k for k in self.__pydantic_extra__.keys()]
    
    @property
    def dimensions(self) -> List[str]:
        # TODO: Remove when dimensions is not accessed any longer
        warnings.warn(
            "Legacy method, will be deprecated soon. This works only if all "
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
        return {k: v for k, v in self.__pydantic_extra__.items()}

    @property
    def dimdict(self) -> Dict[str, List[str]]:
        return {k: v.dimensions for k, v in self.__pydantic_extra__.items()}

    @property
    def var_dim_mapper(self) -> Dict[str, List[str]]:
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
        # TODO: Remove when dimensions is not accessed any longer
        warnings.warn(
            "Legacy method, will be deprecated soon. This works only if all "
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
    def data_variables_min(self):
        return [v.min for v in self.__pydantic_extra__.values()]

    @property
    def data_variables_max(self):
        return [v.max for v in self.__pydantic_extra__.values()]


class Inference(BaseModel):
    model_config = {"validate_assignment" : True}

    objective_function: str = "total_average"
    n_objectives: Annotated[int, to_str] = 1
    objective_names: OptionListStr = []
    backend: Optional[str] = None
    extra_vars: OptionListStr = []
    plot: Optional[str|Callable] = None
    n_predictions: Annotated[int, to_str] = 1

class Multiprocessing(BaseModel):
    _name = "multiprocessing"
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # TODO: Use as private field
    cores: Annotated[int, to_str] = 1
    
    @property
    def n_cores(self) -> Annotated[int, to_str]:
        cpu_avail = mp.cpu_count()
        cpu_set = self.cores
        if cpu_set <= 0:
            return cpu_avail + cpu_set
        else: 
            return cpu_set
        
class Modelparameters(BaseModel):
    __pydantic_extra__: Dict[str,OptionParam]
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @property
    def free(self) -> Dict[str,OptionParam]:
        return {k:v for k, v in self.__pydantic_extra__.items() if v.free}

    @property
    def fixed(self) -> Dict[str,OptionParam]:
        return {k:v for k, v in self.__pydantic_extra__.items() if not v.free}

    @property
    def all(self) -> Dict[str,OptionParam]:
        return self.__pydantic_extra__

    @property
    def n_free(self) -> int:
        return len(self.free)
    
    @property
    def free_value_dict(self) -> Dict[str,float|List[float]]:
        return {k:v.value for k, v in self.free.items()}
    
    @property
    def value_dict(self) -> Dict[str,float|List[float]]:
        return {k:v.value for k, v in self.all.items()}
    

class Errormodel(BaseModel):
    __pydantic_extra__: Dict[str,str]
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    
    @property
    def all(self) -> Dict[str,str]:
        return self.__pydantic_extra__

class Pyabc(BaseModel):
    model_config = {"validate_assignment" : True}

    sampler: str = "SingleCoreSampler"
    population_size: Annotated[int, to_str] = 100
    minimum_epsilon: Annotated[float, to_str] = 0.0
    min_eps_diff: Annotated[float, to_str] = 0.0
    max_nr_populations: Annotated[int, to_str] = 1000
    
    # database configuration
    database_path: str = f"{tempfile.gettempdir()}/pyabc.db"

class Redis(BaseModel):
    model_config = {"validate_assignment" : True, "protected_namespaces": ()}

    _name = "inference.pyabc.redis"

    # redis configuration
    password: str = "nopassword"
    port: Annotated[int, to_str] = 1111

    # eval configuration
    n_predictions: Annotated[int, to_str] = Field(default=50, alias="eval.n_predictions")
    history_id: Annotated[int, to_str] = Field(default=-1, alias="eval.history_id")
    model_id: Annotated[int, to_str] = Field(default=0, alias="eval.model_id")


class Pymoo(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    algortihm: str = "UNSGA3"
    population_size: Annotated[int, to_str] = 100
    max_nr_populations: Annotated[int, to_str] = 1000
    ftol: Annotated[float, to_str] = 1e-5
    xtol: Annotated[float, to_str] = 1e-7
    cvtol: Annotated[float, to_str] = 1e-7
    verbose: Annotated[bool, to_str] = True
    
class Numpyro(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")
    user_defined_probability_model: Optional[str] = None
    user_defined_preprocessing: Optional[str] = None
    gaussian_base_distribution: bool = False
    kernel: str = "nuts"
    init_strategy: str = "init_to_uniform"
    chains: Annotated[int, to_str] = 1
    draws: Annotated[int, to_str] = 2000
    warmup: Annotated[int, to_str] = 1000
    thinning: Annotated[int, to_str] = 1
    svi_iterations: Annotated[int, to_str] = 10_000
    svi_learning_rate: Annotated[float, to_str] = 0.0001
    sa_adapt_state_size: Optional[int] = None



class Config(BaseModel):
    """Configuration manager for pymob."""
    model_config = {"validate_assignment" : True, "extra": "allow", "protected_namespaces": ()}
    _config: configparser.ConfigParser
    _modules: Dict[str,ModuleType]

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
            _cfg_file_paths = _config.read(config)
            _cfg_fp = _cfg_file_paths[0]
        elif isinstance(config, configparser.ConfigParser):
            _config = config
        else:
            _config = configparser.ConfigParser(
                converters=converters,
                interpolation=interp
            )
        # pass arguments to config

        if _cfg_fp is not None: _config.set("case-study", "settings_path", _cfg_fp)
        cfg_dict = {k:dict(s) for k, s in dict(_config).items() if k != "DEFAULT"}
        super().__init__(**cfg_dict)

        self._config = _config
        self._modules = {}

    case_study: Casestudy = Field(default=Casestudy(), alias="case-study")
    simulation: Simulation = Field(default=Simulation())
    data_structure: Datastructure = Field(default=Datastructure(), alias="data-structure") # type:ignore
    inference: Inference = Field(default=Inference())
    model_parameters: Modelparameters = Field(default=Modelparameters(), alias="model-parameters") #type: ignore
    error_model: Errormodel = Field(default=Errormodel(), alias="error-model") # type: ignore
    multiprocessing: Multiprocessing = Field(default=Multiprocessing())
    inference_pyabc: Pyabc = Field(default=Pyabc(), alias="inference.pyabc")
    inference_pyabc_redis: Redis = Field(default=Redis(), alias="inference.pyabc.redis")
    inference_pymoo: Pymoo = Field(default=Pymoo(), alias="inference.pymoo")
    inference_numpyro: Numpyro = Field(default=Numpyro(), alias="inference.numpyro")
        
    @property
    def input_file_paths(self) -> list:
        paths_input_files = []
        for file in self.simulation.input_files:
            fp = scenario_file(file, self.case_study.name, self.case_study.scenario)
            paths_input_files.append(fp)


        for file in self.case_study.observations:
            if not os.path.isabs(file):
                fp = os.path.join(self.case_study.data_path, file)
            else:
                fp = file
            paths_input_files.append(fp)

        return paths_input_files

    def print(self):
        print("Simulation configuration", end="\n")
        print("========================")
        for section, field_info in self.model_fields.items():
            print(f"{section}({getattr(self, section)})", end="\n") # type: ignore

        print("========================", end="\n")

    def save(self, fp: Optional[str]=None, force=False):
        """Saves the configuration to a settings.cfg file
        
        Uses serializers defined at the top, which parse the options to str
        so they can be processed by configfile. 

        In case the model configuration should be stored to a json file use
        something like `json.dumps(self.model_dump())`, because the build in
        function, is somewhat disabled by the listparsers which are needed for
        configfile lists.

        Parameters
        ----------
        fp: Optiona[str] file path to write the settings file to
        force: [bool] should the settings file be overwritten without asking
            for user confirmation (default: False)
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

        if write:
            with open(file_path, "w") as f:
                self._config.write(f)

    def create_directory(self, directory: Literal["results", "scenario"], force=False):
        if directory == "results":
            p = os.path.abspath(self.case_study.output_path)
        elif directory == "scenario":
            p = os.path.abspath(self.case_study.scenario_path)
        else:
            raise NotImplementedError(
                f"{directory.capitalize()} is not an expected directory in the "
                "case study logic. Use one of 'results' or 'scenario'."
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

    def import_casestudy_modules(self):
        """
        this script handles the import of a case study without the typical 
        __init__.py file. It iterates through all .py files in the root directory
        of the case study (typically: sim, mod, stats, plot, data, prior)
        and imports them with import_module(...)
        """

        # append relevant paths to sys
        package = os.path.join(
            self.case_study.root, 
            self.case_study.package
        )
        if package not in sys.path:
            sys.path.append(package)
            print(f"Appended '{package}' to PATH")
    
        case_study = os.path.join(
            self.case_study.root, 
            self.case_study.package,
            self.case_study.name
        )
        if case_study not in sys.path:
            sys.path.append(case_study)
            print(f"Appended '{case_study}' to PATH")

        for module in self.case_study.modules:
            try:
                m = importlib.import_module(module, package=case_study)
                self._modules.update({module: m})
            except ModuleNotFoundError:
                warnings.warn(
                    f"Module {module}.py not found in {case_study}."
                    f"Missing modules can lead to unexpected behavior."
                    "If a module is not imported, you can specify it in the "
                    "Config 'config.case_study.modules = [...]'"
                )

    def import_simulation_from_case_study(self) -> Callable:
        try:
            Simulation = getattr(self._modules["sim"], self.case_study.simulation)
        except:
            raise ImportError(
                f"Simulation class {self.case_study.simulation} "
                "could not be found. Make sure the simulaton option is spelled "
                "correctly or specify an class that exists in sim.py"
            )
        
        return Simulation
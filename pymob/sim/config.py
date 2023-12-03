import os
import configparser
import warnings
import multiprocessing as mp
from typing import List, Optional, Union, Dict, Any
from typing_extensions import Annotated
import tempfile

import numpy as np

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic.functional_validators import BeforeValidator, AfterValidator
from pydantic.functional_serializers import PlainSerializer

from pymob.utils.store_file import scenario_file, converters

def string_to_list(option: Union[List, str]) -> List:
    if isinstance(option, (list, tuple)):
        return list(option)
    
    if " " not in option:
        return [option] 
    else:
        return [i.strip() for i in option.split(" ")]


def list_to_string(lst: List):
    return " ".join([str(l) for l in lst])


serialize_list_to_string = PlainSerializer(
    list_to_string, 
    return_type=str, 
    when_used="json"
)

to_str = PlainSerializer(lambda x: str(x), return_type=str, when_used="json")

OptionListStr = Annotated[
    List[str], 
    BeforeValidator(string_to_list), 
    serialize_list_to_string
]

OptionListFloat = Annotated[
    List[float], 
    BeforeValidator(string_to_list), 
    serialize_list_to_string
]


class FloatParam(BaseModel):
    name: str
    value: Optional[float]
    min: Optional[float]
    max: Optional[float]
    step: Optional[float]
    prior: Optional[str]


        
class Casestudy(BaseModel):
    model_config = {"validate_assignment" : True}

    name: str = "unnamed_case_study"
    scenario: str = "unnamed_scenario"
    
    output: str = "."
    data: str = "."
    logging: str = "DEBUG"
    observations: OptionListStr = []
    package: str = "case_studies"
    root: Optional[str] = os.getcwd()
    settings_path: Optional[str] = Field(default=None, exclude=True)

    @computed_field
    @property
    def output_path(self) -> str:
        if not os.path.isabs(self.output):
            return os.path.join(
                os.path.relpath(self.output),
                os.path.relpath(self.package),
                os.path.relpath(self.name),
                "results",
                self.scenario,
            )
        else:
            return os.path.abspath(self.output)

    @computed_field
    @property
    def data_path(self) -> str:
        if not os.path.isabs(self.data):
            return os.path.join(
                self.package, 
                os.path.relpath(self.name),
                os.path.relpath(self.data)
            ) 
        else:
            return os.path.abspath(self.data)
    
    @computed_field
    @property
    def settings(self) -> str:
        if self.settings_path is not None:
            return self.settings_path
        else:
            return os.path.join(
                os.path.relpath(self.package),
                os.path.relpath(self.name),
                "scenarios", 
                self.scenario, 
                "settings.cfg"
            )
        

class Simulation(BaseModel):
    model_config = {"validate_assignment" : True, "extra": "allow"}

    input_files: OptionListStr = []
    dimensions: OptionListStr = []
    data_variables: OptionListStr = []
    seed: Annotated[int, to_str] = 1
    data_variables_min: Optional[OptionListFloat] = Field(default=None, validate_default=True)
    data_variables_max: Optional[OptionListFloat] = Field(default=None, validate_default=True)

    @model_validator(mode='after')
    def post_update(self):
        if self.data_variables_min is not None:
            if len(self.data_variables_min) != len(self.data_variables):
                self.data_variables_min = None
        
        if self.data_variables_max is not None:
            if len(self.data_variables_max) != len(self.data_variables):
                self.data_variables_max = None
                
        return self
        
    @field_validator("data_variables_min", "data_variables_max", mode="after")
    def set_data_variable_bounds(cls, v, info, **kwargs):
        # For conditionally updating values (e.g. when data variables change)
        # see https://github.com/pydantic/pydantic/discussions/7127
        data_variables = info.data.get("data_variables")
        if v is not None:
            if len(v) != len(data_variables):
                raise AssertionError(
                    "If bounds are provided, the must be provided for all data "
                    "variables. If a bound for a variable is unknown, write 'nan' "
                    "in the config file at the position of the variable. "
                    "\nE.g.:"
                    "\ndata_variables = A B C"
                    "\ndata_variables_max = 4 nan 2"
                    "\ndata_variables_min = 0 0 nan"
                )
            else:
                return v
        else:
            return [float("nan")] * len(data_variables)
    
class Inference(BaseModel):
    model_config = {"validate_assignment" : True}

    objective_function: str = "total_average"
    n_objectives: Annotated[int, to_str] = 1
    objective_names: OptionListStr = []
    backend: Optional[str] = None

class Multiprocessing(BaseModel):
    _name = "multiprocessing"
    model_config = {"validate_assignment" : True, "extra": "ignore"}

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
    model_config = {"validate_assignment" : True, "extra": "allow"}


class Pyabc(BaseModel):
    model_config = {"validate_assignment" : True}

    sampler: str = "SingleCoreSampler"
    population_size: Annotated[int, to_str] = 100
    minimum_epsilon: Annotated[float, to_str] = 0.0
    min_eps_diff: Annotated[float, to_str] = 0.0
    max_nr_populations: Annotated[int, to_str] = 1000
    plot_function: Optional[str] = None
    
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
    model_config = {"validate_assignment" : True}

    algortihm: str = "UNSGA3"
    population_size: Annotated[int, to_str] = 100
    max_nr_populations: Annotated[int, to_str] = 1000
    ftol: Annotated[float, to_str] = 1e-5
    xtol: Annotated[float, to_str] = 1e-7
    cvtol: Annotated[float, to_str] = 1e-7
    verbose: Annotated[bool, to_str] = True
    

class Config(BaseModel):
    model_config = {"validate_assignment" : True, "extra": "allow", "protected_namespaces": ()}
    _config: configparser.ConfigParser

    def __init__(
        self,
        config: Optional[Union[str, configparser.ConfigParser]],
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

    case_study: Casestudy = Field(default=Casestudy(), alias="case-study")
    simulation: Simulation = Field(default=Simulation())
    inference: Inference = Field(default=Inference())
    model_parameters: Modelparameters = Field(default=Modelparameters(), alias="model-parameters")
    multiprocessing: Multiprocessing = Field(default=Multiprocessing())
    inference_pyabc: Pyabc = Field(default=Pyabc(), alias="inference.pyabc")
    inference_pyabc_redis: Redis = Field(default=Redis(), alias="inference.pyabc.redis")
    inference_pymoo: Pymoo = Field(default=Pymoo(), alias="inference.pymoo")
        
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

    def save(self):
        """Saves the configuration to a settings.cfg file
        
        Uses serializers defined at the top, which parse the options to str
        so they can be processed by configfile. 

        In case the model configuration should be stored to a json file use
        something like `json.dumps(self.model_dump())`, because the build in
        function, is somewhat disabled by the listparsers which are needed for
        configfile lists.
        """

        settings = self.model_dump(
            by_alias=True, 
            mode="json", 
            exclude_none=True,
            exclude={"case_study": {"settings", "output_path", "data_path", "root"}}
        )
        self._config.update(**settings)

        with open(self.case_study.settings, "w") as fp:
            self._config.write(fp)
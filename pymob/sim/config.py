import os
import configparser
import warnings
import logging
import multiprocessing as mp
from typing import List, Optional, Union, Dict, Any, Literal, Callable
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
    
    if len(option) == 0:
        return []
    elif " " not in option:
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
    init_root: str = Field(default=os.getcwd(), exclude=True)
    root: str = "."

    name: str = "unnamed_case_study"
    scenario: str = "unnamed_scenario"
    package: str = "case_studies"
    
    output: Optional[str] = None
    data: Optional[str] = None

    observations: OptionListStr = []
    
    logging: str = "DEBUG"

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
    dimensions: OptionListStr = []
    data_variables: OptionListStr = []
    n_ode_states: int = -1
    replicated: bool = False
    modeltype: Literal["stochastic", "deterministic"] = "stochastic"
    solver_post_processing: Optional[str] = Field(default=None, validate_default=True)
    seed: Annotated[int, to_str] = 1
    data_variables_min: Optional[OptionListFloat] = Field(default=None, validate_default=True)
    data_variables_max: Optional[OptionListFloat] = Field(default=None, validate_default=True)
    evaluator_dim_order: OptionListStr = []

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
    extra_vars: OptionListStr = []
    plot_function: Optional[str] = None
    n_predictions: Annotated[int, to_str] = 1

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
    
class Numpyro(BaseModel):
    model_config = {"validate_assignment" : True}
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
    model_parameters: Modelparameters = Field(default=Modelparameters(), alias="free-model-parameters")
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
            exclude={"case_study": {"output_path", "data_path", "root", "init_root"}}
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
import os
import configparser
import multiprocessing as mp
from typing import List, Optional, Union, Dict, Any
from typing_extensions import Annotated

import numpy as np

from pydantic import BaseModel, Field, computed_field, validator, model_validator
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
    return " ".join(lst)


serialize_list_to_string = PlainSerializer(
    list_to_string, 
    return_type=str, 
    when_used="always"
)


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


class Config:
    def __init__(
        self,
        config: Optional[Union[str, configparser.ConfigParser]],
    ) -> None:
  
        
        if isinstance(config, str):
            self._config = configparser.ConfigParser(converters=converters)        
            self._config.read(config)
        elif isinstance(config, configparser.ConfigParser):
            self._config = config
        else:
            self._config = configparser.ConfigParser(converters=converters)

        # load config sections with pydantic
        self.case_study = self.CasestudySect(**self.load_section("case-study"))
        self.simulation = self.SimulationSect(**self.load_section("simulation"))
        self.inference = self.InferenceSect(**self.load_section("inference"))
        self.multiprocessing = self.MultiprocessingSect(**self.load_section("multiprocessing"))
        self.model_parameters = self.ModelparameterSect(**self.load_section("model-parameters"))


        self.print()
        
    class CasestudySect(BaseModel):
        _name = "case-study"
        model_config = {"validate_assignment" : True}

        name: str = "unnamed_case_study"
        scenario: str = "unnamed_scenario"
        
        output: str = "."
        data: str = "."
        logging: str = "DEBUG"
        observations: OptionListStr = []
        package: str = "case_studies"
        root: Optional[str] = None

        @computed_field
        @property
        def output_path(self) -> str:
            if not os.path.isabs(self.output):
                return os.path.join(
                    os.path.relpath(self.output),
                    os.path.relpath(self.package),
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
                    os.path.relpath(self.data)
                ) 
            else:
                return os.path.abspath(self.data)
        

    class SimulationSect(BaseModel):
        _name = "simulation"
        model_config = {"validate_assignment" : True, "extra": "allow"}

        input_files: OptionListStr = []
        dimensions: OptionListStr = []
        data_variables: OptionListStr = []
        seed: int = 1
        data_variables_min: Optional[OptionListFloat] = None
        data_variables_max: Optional[OptionListFloat] = None

        @model_validator(mode='after')
        def post_update(self) -> Dict[str, Any]:
            if len(self.data_variables_min) != len(self.data_variables):
                self.data_variables_min = None
                self.data_variables_max = None
            # else:
            #     values['last_enabled'] = None
            #     values['last_disabled'] = datetime.now()

            return self
            
        @validator("data_variables_min", "data_variables_max", always=True)
        def set_data_variable_bounds(cls, v, values, **kwargs):
            # For conditionally updating values (e.g. when data variables change)
            # see https://github.com/pydantic/pydantic/discussions/7127
            data_variables = values.get("data_variables")
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
        
    class InferenceSect(BaseModel):
        _name = "inference"
        model_config = {"validate_assignment" : True}

        objective_function: str = "total_average"
        n_objectives: int = 1
        objective_names: List = []
        backend: Optional[str] = None

    class MultiprocessingSect(BaseModel):
        _name = "multiprocessing"
        model_config = {"validate_assignment" : True, "extra": "ignore"}

        # TODO: Use as private field
        cores: int = 1
        
        @computed_field
        @property
        def n_cores(self) -> int:
            cpu_avail = mp.cpu_count()
            cpu_set = self.cores
            if cpu_set <= 0:
                return cpu_avail + cpu_set
            else: 
                return cpu_set
            
    class ModelparameterSect(BaseModel):
        _name = "model-parameters"
        model_config = {"validate_assignment" : True, "extra": "allow"}

    def load_section(self, section: str) -> dict:
        if self._config.has_section(section):
            return dict(self._config[section])
        else:
            return {}

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
    
    @property
    def sections(self) -> List[BaseModel]:
        return [i for _, i in self.__dict__.items() if isinstance(i, BaseModel)]
    
    def print(self):
        print("Simulation configuration", end="\n")
        print("========================")
        for section in self.sections:
            print(f"{section._name}({section})", end="\n") # type: ignore

        print("========================", end="\n")

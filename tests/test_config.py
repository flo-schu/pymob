import pytest
import tempfile
from pymob.simulation import SimulationBase, Config
from pymob.sim.config import ArrayParam, FloatParam, DataVariable, Datastructure
from pymob.utils.store_file import import_package
import xarray as xr
import os

scenario = "case_studies/test_case_study/scenarios/test_scenario_scripting_api"

def test_simulation():
    sim = SimulationBase()
    sim.config.case_study.name = "test_case_study"
    sim.config.case_study.scenario = "test_scenario_scripting_api"
    sim.config.case_study.observations = ["simulated_data.nc"]
    sim.config.case_study.data_path = None

    # load data before specifying dims
    sim.config.case_study.data = os.path.abspath("case_studies/test_case_study/data")
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    

    # try wrong specification
    threw_error = None
    try:
        sim.config.data_structure.rabbits = DataVariable(dimensions=["hash"])
        sim.config.data_structure.wolves = DataVariable(dimensions=["time"])
        sim.observations = xr.load_dataset(sim.config.input_file_paths[0])
        threw_error = False
    except KeyError:
        threw_error = True

    assert threw_error

    sim.validate()

    sim.config.data_structure.rabbits = DataVariable(dimensions=["time"])
    sim.config.data_structure.wolves = DataVariable(dimensions=["time"])
    
    # load data by providing an absolute path
    sim.config.case_study.data = os.path.abspath("case_studies/test_case_study/data")
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    

    # load data by providing a relative path
    sim.config.case_study.data = "case_studies/test_case_study/data"
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    
    
    # load data by providing no path (the default 'data' directory in the case study)
    sim.config.case_study.data = None
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    

    sim.config.case_study.output = None

    sim.setup()
    sim.config.save(
        fp=f"{scenario}/test_settings.cfg",
        force=True, 
    )

def test_load_generated_settings():
    sim = SimulationBase(f"{scenario}/test_settings.cfg")
    assert sim.config.case_study.name == "test_case_study"
    assert sim.config.case_study.scenario == "test_scenario_scripting_api"
    assert sim.config.case_study.package == "case_studies"
    assert sim.config.case_study.data == None
    assert sim.config.case_study.data_path == "case_studies/test_case_study/data"
    assert sim.config.case_study.output == None
    assert sim.config.case_study.output_path == \
        "case_studies/test_case_study/results/test_scenario_scripting_api"

def test_load_interpolated_settings():
    sim = SimulationBase(f"{scenario}/interp_settings.cfg")
    expected_output = \
        "./case_studies/test_case_study/results/test_scenario_scripting_api"
    assert sim.config.case_study.output == expected_output



def test_standalone_casestudy():
    wd = os.getcwd()
    case_study_name = "test_case_study_standalone"
    root = os.path.join(str(tempfile.tempdir), case_study_name)
    os.mkdir(root)
    os.chdir(root)
    
    # this is the syntax for setting up a standalone case study
    # currently root cannot be set with the config backend, but needs
    # to be specified with `chdir`
    sim = SimulationBase()
    sim.config.case_study.name = "."
    sim.config.case_study.scenario = "test_scenario_standalone"
    sim.config.case_study.package = "."

    os.makedirs(sim.config.case_study.output_path)
    os.makedirs(sim.config.case_study.data_path)
    os.makedirs(sim.config.case_study.scenario_path)
    sim.config.save(force=True)

    # test if all files exist and remove test directory
    os.chdir(wd)
    file_structure = [
        f"{tempfile.tempdir}/test_case_study_standalone",
        f"{tempfile.tempdir}/test_case_study_standalone/data",
        f"{tempfile.tempdir}/test_case_study_standalone/results",
        f"{tempfile.tempdir}/test_case_study_standalone/results/test_scenario_standalone",
        f"{tempfile.tempdir}/test_case_study_standalone/scenarios",
        f"{tempfile.tempdir}/test_case_study_standalone/scenarios/test_scenario_standalone",
        f"{tempfile.tempdir}/test_case_study_standalone/scenarios/test_scenario_standalone/settings.cfg",
    ]
    
    for p in reversed(file_structure):
        assert os.path.exists(p)
        if os.path.isdir(p):
            os.rmdir(p)
        else:
            os.remove(p)

def test_parameter_parsing():
    config = Config()

    io = "value=1.0 min=0.0 max=3.0 free=True"

    # test scripting input
    test = FloatParam(value=1.0, min=0.0, max=3.0)
    config.model_parameters.test = test

    # test dict input
    config.model_parameters.test = test.model_dump(exclude_none=True)
    assert config.model_parameters.test == test # type: ignore

    # test validation
    config.model_parameters.test = io
    assert config.model_parameters.test == test # type: ignore

    # test serialization
    serialized = config.model_parameters.model_dump(mode="json")
    assert serialized == {"test": io}


def test_parameter_array():
    config = Config()

    io = "value=[1.0,2.0,3.0] free=True"

    # test scripting input
    test = ArrayParam(value=[1.0,2.0,3.0])
    config.model_parameters.test = test

    # test dict input
    config.model_parameters.test = test.model_dump(exclude_none=True)
    assert config.model_parameters.test == test # type: ignore

    # test config file input
    config.model_parameters.test = io
    assert config.model_parameters.test == test  # type: ignore
    
    # test serialization
    serialized = config.model_parameters.model_dump(mode="json")
    assert serialized == {"test": io}


def test_model_parameters():
    config = Config()

    a = FloatParam(value=1)
    b = FloatParam(value=5, free=False)

    config.model_parameters.a = a
    config.model_parameters.b = b

    frmp = config.model_parameters.free
    fimp = config.model_parameters.fixed
    almp = config.model_parameters.all

    assert frmp == {"a": a}
    assert fimp == {"b": b}
    assert almp == {"a": a, "b":b}

def test_error_model():
    config = Config()

    a = "lognorm(loc=1,scale=2)"
    config.error_model.a = a

    
def test_data_variables():
    config = Config()
    config.case_study.name = "test_case_study"
    config.case_study.scenario = "test_scenario_scripting_api"
    config.data_structure.wolves = DataVariable(dimensions=["time"], min=0)
    assert config.data_structure.data_variables == ["wolves"]
    config.save(force=True)
    
    config.data_structure = {"wolves": dict(dimensions = ["time"])} # type: ignore
    assert config.data_structure.data_variables == ["wolves"]


    config.data_structure.B = DataVariable(dimensions=["a", "b"], dimensions_evaluator=["b","a"])
    assert config.data_structure.dimdict == {"wolves": ["time"], "B": ["a", "b"]}
    assert config.data_structure.var_dim_mapper == {"wolves": [0], "B": [1,0]}


if __name__ == "__main__":
    # test_data_variables()
    pass
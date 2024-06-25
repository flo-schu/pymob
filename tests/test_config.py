import pytest
import tempfile
from pymob.simulation import SimulationBase
from pymob.utils.store_file import import_package
import xarray as xr
import os

scenario = "case_studies/test_case_study/scenarios/test_scenario_scripting_api"

def test_simulation():
    sim = SimulationBase()
    sim.config.case_study.name = "test_case_study"
    sim.config.case_study.scenario = "test_scenario_scripting_api"
    sim.config.case_study.observations = ["simulated_data.nc"]
    sim.config.simulation.data_variables = ["rabbits", "wolves"]
    sim.config.simulation.dimensions = ["time"]
    
    sim.validate()

    
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

if __name__ == "__main__":
    # test_simulation()
    pass
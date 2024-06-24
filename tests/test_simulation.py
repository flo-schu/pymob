import pytest
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


if __name__ == "__main__":
    # test_simulation()
    test_load_generated_settings()
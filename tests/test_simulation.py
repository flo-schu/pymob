import pytest
from pymob.simulation import SimulationBase
from pymob.utils.store_file import import_package
import xarray as xr

test_settings = "case_studies/test_case_study/scenarios/test_scenario_scripting_api/test_settings.cfg"

def test_simulation():
    sim = SimulationBase()

    sim.config.case_study.name = "test_case_study"
    sim.config.case_study.scenario = "test_scenario_scripting_api"
    sim.config.case_study.package = "case_studies"
    sim.config.case_study.data = "case_studies/test_case_study/data"
    sim.config.case_study.output = "case_studies/test_case_study/results/test_scenario_scripting_api"
    
    sim.config.case_study.observations = ["simulated_data.nc"]
    sim.config.simulation.data_variables = ["rabbits", "wolves"]
    sim.config.simulation.dimensions = ["time"]
    sim.config.case_study.settings_path = test_settings
    sim.validate()

    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    
    sim.setup()
    sim.config.save(force=True)

def test_load_interpolated_settings():
    sim = SimulationBase("case_studies/test_case_study/scenarios/test_scenario_scripting_api/interp_settings.cfg")
    expected_output = "./case_studies/test_case_study/results/test_scenario_scripting_api"
    assert sim.config.case_study.output == expected_output

def test_load_generated_settings():
    sim = SimulationBase("case_studies/test_case_study/scenarios/test_scenario_scripting_api/test_settings.cfg")
    expected_output = "case_studies/test_case_study/results/test_scenario_scripting_api"
    assert sim.config.case_study.output == expected_output

if __name__ == "__main__":
    test_simulation()
    test_load_generated_settings()
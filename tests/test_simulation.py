import pytest
from pymob.simulation import SimulationBase
from pymob.utils.store_file import import_package
import xarray as xr
import os

scenario = "case_studies/test_case_study/scenarios/test_scenario_scripting_api"

def test_simulation():
    sim = SimulationBase()
    os.chdir("case_studies/test_case_study/")
    
    sim.config.case_study.data = "data"
    sim.config.case_study.output = "results/test_scenario_scripting_api"
    sim.config.case_study.observations = ["simulated_data.nc"]
    sim.config.simulation.data_variables = ["rabbits", "wolves"]
    sim.config.simulation.dimensions = ["time"]
    sim.config.case_study.settings_path = \
        "scenarios/test_scenario_scripting_api/test_settings.cfg"
    
    sim.validate()

    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    
    sim.setup()
    sim.config.save(force=True)

def test_load_interpolated_settings():
    sim = SimulationBase(f"{scenario}/interp_settings.cfg")
    expected_output = \
        "./case_studies/test_case_study/results/test_scenario_scripting_api"
    assert sim.config.case_study.output == expected_output

def test_load_generated_settings():
    sim = SimulationBase(f"{scenario}/test_settings.cfg")
    expected_output = "results/test_scenario_scripting_api"
    assert sim.config.case_study.output == expected_output

if __name__ == "__main__":
    pass
    # test_simulation()
    # test_load_generated_settings()
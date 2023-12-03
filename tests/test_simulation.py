import pytest
from pymob.simulation import SimulationBase
from pymob.utils.store_file import import_package
import xarray as xr

def test_simulation():
    # pytest.skip()
    # sim = SimulationBase("case_studies/test_case_study/scenarios/test_scenario/settings.cfg")
    sim = SimulationBase()

    sim.config.case_study.name = "test_case_study"
    sim.config.case_study.scenario = "test_scenario_scripting_api"
    sim.config.case_study.package = "case_studies"
    sim.config.case_study.data = "data"
    sim.config.case_study.observations = ["simulated_data.nc"]
    sim.config.simulation.data_variables = ["rabbits", "wolves"]
    sim.config.simulation.dimensions = ["time"]
    sim.config.case_study.settings_path = "./settings.cfg"
    sim.validate()

    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])
    

    sim.setup()
    sim.config.save()


if __name__ == "__main__":
    test_simulation()
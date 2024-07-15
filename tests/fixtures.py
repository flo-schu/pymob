import numpy as np
import pytest

from pymob.sim.config import Config, DataVariable, FloatParam
from pymob.simulation import SimulationBase
from pymob.utils.store_file import prepare_casestudy

rng = np.random.default_rng(1)

def init_simulation_casestudy_api(scenario="test_scenario"):
    config = prepare_casestudy(
        case_study=("test_case_study", scenario),
        config_file="settings.cfg",
        pkg_dir="case_studies"
    )
    
    from case_studies.test_case_study.sim import Simulation
    sim = Simulation(config=config)
    sim.setup()
    return sim

def init_simulation_scripting_api(scenario="test_scenario"):
    config = Config()

    config.case_study.name = "test_case_study"
    config.case_study.scenario = scenario
    config.case_study.package = "case_studies"
    config.case_study.simulation = "Simulation"
    config.import_casestudy_modules(reset_path=True)

    Simulation = config.import_simulation_from_case_study()
    sim = Simulation(config)
    sim.setup()



def init_bufferguts_casestudy(scenario="testing"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinator-tktd/case_studies"
    config.case_study.simulation = "BuffergutsSimulation"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()


def init_simulation_scripting_api_v2(scenario="test_scenario"):
    config = Config()

    config.case_study.name = "test_case_study"
    config.case_study.scenario = scenario
    config.case_study.package = "case_studies"
    config.case_study.simulation = "Simulation"
    config.import_casestudy_modules(reset_path=True)

    # and this works as well,
    from test_case_study.sim import Simulation # type: ignore
    sim = Simulation(config)
    sim.setup()


def linear_model():
    def model(x, a, b):
        return a + x * b

    parameters = dict(
        a=0,
        b=1,
        sigma_y=1,
    )

    x = np.linspace(-5, 5, 50)
    y = model(x=x, a=parameters["a"], b=parameters["b"])
    y_noise = rng.normal(loc=y, scale=parameters["sigma_y"])

    return model, x, y, y_noise, parameters
from pymob.sim.config import Config
from pymob.simulation import SimulationBase
from pymob.utils.store_file import prepare_casestudy

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
    config.import_casestudy_modules()

    Simulation = config.import_simulation_from_case_study()
    sim = Simulation(config)
    sim.setup()


def init_simulation_scripting_api_v2(scenario="test_scenario"):
    config = Config()

    config.case_study.name = "test_case_study"
    config.case_study.scenario = scenario
    config.case_study.package = "case_studies"
    config.case_study.simulation = "Simulation"
    config.import_casestudy_modules()

    # and this works as well,
    from test_case_study.sim import Simulation # type: ignore
    sim = Simulation(config)
    sim.setup()
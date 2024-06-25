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
    pass
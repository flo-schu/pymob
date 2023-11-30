from pymob.utils.store_file import prepare_casestudy

def init_test_case_study():
    config = prepare_casestudy(
        case_study=("test_case_study", "test_scenario"),
        config_file="settings.cfg",
        pkg_dir="case_studies"
    )
    
    from case_studies.test_case_study.sim import Simulation
    
    sim = Simulation(config=config)
    sim.setup()

    return sim
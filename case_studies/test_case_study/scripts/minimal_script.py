from pymob.utils.store_file import prepare_casestudy

config = prepare_casestudy(
    case_study=("test_case_study", "test_scenario"), 
    config_file="settings.cfg", 
    # use this to navigate to the directory where your case_studies
    # are collected relative to your working directory
    pkg_dir="case_studies"
)

from sim import Simulation

sim = Simulation(config)


f = sim.dispatch(theta={"alpha": 0.5})  # initiate the simulation
f()  # run the simulation (takes more time for the first run)
f.results  # obtain the results

sim.plot(f.results)

print(config)
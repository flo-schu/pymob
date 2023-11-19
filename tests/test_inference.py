import subprocess
import xarray as xr
import numpy as np
from click.testing import CliRunner

from pymob.utils.store_file import prepare_casestudy

def test_scripting_API():
    config = prepare_casestudy(
        case_study=("test_case_study", "test_scenario"),
        config_file="settings.cfg"
    )
    from case_studies.test_case_study.sim import Simulation
    
    sim = Simulation(config=config)
    sim.set_inferer(backend="pyabc")
    sim.inferer.run()

    # TODO: write test (something like if error smaller x)


def test_inference_evaluation():
    config = prepare_casestudy(
        case_study=("test_case_study", "test_scenario"),
        config_file="settings.cfg"
    )
    from case_studies.test_case_study.sim import Simulation
    
    sim = Simulation(config=config)
    sim.set_inferer(backend="pyabc")

    sim.inferer.load_results()
    fig = sim.inferer.plot_chains()
    fig.savefig(sim.output_path + "/pyabc_chains.png")
    ax = sim.inferer.plot_predictions(
        data_variable="rabbits", 
        x_dim="time"
    )
    fig = ax.get_figure()
    fig.savefig(sim.output_path + "/pyabc_posterior_predictions.png")


def test_commandline_API_redis():
    from pymob.infer import main
    runner = CliRunner()
    
    args = "--case_study=test_case_study --scenario=test_scenario"
    result = runner.invoke(main, args.split(" "))

def test_commandline_API_infer():
    from pymob.infer import main
    runner = CliRunner()
    
    args = "--case_study=test_case_study --scenario=test_scenario"
    result = runner.invoke(main, args.split(" "))


# test_inference_evaluation()
test_commandline_API_infer()
import subprocess
import xarray as xr
import numpy as np
from click.testing import CliRunner

from tests.fixtures import init_test_case_study


def test_pyabc():
    sim = init_test_case_study()
    sim.set_inferer(backend="pyabc")
    sim.inferer.run()

    # TODO: write test (something like if error smaller x)


def test_pymoo():
    sim = init_test_case_study()
    sim.set_inferer(backend="pymoo")
    sim.inferer.run()

    # TODO: write test (something like if error smaller x)


def test_inference_evaluation():
    sim = init_test_case_study()
    sim.set_inferer(backend="pyabc")

    sim.inferer.load_results()
    fig = sim.inferer.plot_chains() # type: ignore
    fig.savefig(sim.config.case_study.output_path + "/pyabc_chains.png")
    ax = sim.inferer.plot_predictions(
        data_variable="rabbits", 
        x_dim="time"
    )
    fig = ax.get_figure()
    fig.savefig(sim.config.case_study.output_path + "/pyabc_posterior_predictions.png")


def test_commandline_API_infer():
    from pymob.infer import main
    runner = CliRunner()
    
    args = "--case_study=test_case_study --scenario=test_scenario"
    result = runner.invoke(main, args.split(" "))


# test_inference_evaluation()
# test_commandline_API_infer()

if __name__ == "__main__":
    import sys; import os; sys.path.append(os.getcwd())
    test_pymoo()
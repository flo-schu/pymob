import json
import xarray as xr
import numpy as np
from click.testing import CliRunner
from matplotlib import pyplot as plt

from tests.fixtures import init_test_case_study


def test_scripting_api_pyabc():
    sim = init_test_case_study()
    sim.set_inferer(backend="pyabc")
    sim.inferer.run()
    sim.inferer.load_results()
    
    posterior_mean = sim.inferer.idata.posterior.mean(("chain", "draw"))
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=5e-2, atol=1e-5
    )


def test_pymoo():
    sim = init_test_case_study()
    sim.set_inferer(backend="pymoo")
    sim.inferer.run()

    with open(f"{sim.config.case_study.output_path}/pymoo_params.json", "r") as f:
        pymoo_results = json.load(f)

    estimated_parameters = pymoo_results["X"]
    true_parameters = sim.model_parameter_dict
    
    np.testing.assert_allclose(
        np.array(list(estimated_parameters.values())),
        np.array(list(true_parameters.values())),
        rtol=5e-2, atol=1e-5
    )


    # TODO: write test (something like if error smaller x)


def test_inference_evaluation():
    sim = init_test_case_study()
    sim.set_inferer(backend="pyabc")

    sim.inferer.load_results()
    fig = sim.inferer.plot_chains()
    fig.savefig(sim.output_path + "/pyabc_chains.png")

    # posterior predictions
    for data_var in sim.data_variables:
        ax = sim.inferer.plot_predictions(
            data_variable=data_var, 
            x_dim="time"
        )
        fig = ax.get_figure()

        fig.savefig(f"{sim.output_path}/pyabc_posterior_predictions_{data_var}.png")
        plt.close()

def test_commandline_API_infer():
    from pymob.infer import main
    runner = CliRunner()
    
    args = "--case_study=test_case_study --scenario=test_scenario"
    result = runner.invoke(main, args.split(" "))

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_scripting_api_pyabc()
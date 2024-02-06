import subprocess
import xarray as xr
import numpy as np
from click.testing import CliRunner
from matplotlib import pyplot as plt

from pymob.utils.store_file import prepare_casestudy

def create_simulation():
    config = prepare_casestudy(
        case_study=("test_case_study", "test_scenario"),
        config_file="settings.cfg"
    )
    from case_studies.test_case_study.sim import Simulation
    return Simulation(config=config)

def test_scripting_api_pyabc():
    sim = create_simulation()
    sim.set_inferer(backend="pyabc")
    sim.inferer.run()
    sim.inferer.store_results()
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


def test_inference_evaluation():
    sim = create_simulation()
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

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
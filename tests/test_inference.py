import pytest
from matplotlib import pyplot as plt
from pymob.solvers import JaxSolver
import numpy as np

from tests.fixtures import (
    init_simulation_casestudy_api,
    init_test_case_study_hierarchical_presimulated
)


def test_inference_evaluation():
    pytest.skip()
    sim = init_simulation_casestudy_api()
    sim.set_inferer(backend="pyabc")

    sim.inferer.load_results()
    fig = sim.inferer.plot_chains()
    fig.savefig(sim.output_path + "/pyabc_chains.png")

    # posterior predictions
    for data_var in sim.config.data_structure.data_variables:
        ax = sim.inferer.plot_predictions(
            data_variable=data_var, 
            prediction_data_variable=data_var,
            x_dim="time"
        )
        fig = ax.get_figure()

        fig.savefig(f"{sim.output_path}/pyabc_posterior_predictions_{data_var}.png")
        plt.close()



def test_posterior():
    sim = init_test_case_study_hierarchical_presimulated("lotka_volterra_hierarchical_presimulated_v1")

    sim.config.inference_numpyro.chains = 1
    sim.config.inference_numpyro.draws = 5
    sim.config.inference_numpyro.nuts_max_tree_depth = 1
    sim.config.inference_numpyro.warmup = 0
    sim.config.inference_numpyro.gaussian_base_distribution = True
    sim.config.inference_numpyro.svi_iterations = 50
    sim.config.inference_numpyro.kernel = "svi"

    sim.solver = JaxSolver
    sim.dispatch_constructor()
    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    idata = sim.inferer.idata

    np.testing.assert_array_equal(idata.posterior.coords["id"], sim.observations["id"])
    np.testing.assert_array_equal(idata.posterior.coords["rabbit_species"], sim.dimension_coords["rabbit_species"])
    np.testing.assert_array_equal(idata.posterior.coords["experiment"], sim.dimension_coords["experiment"])


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_scripting_api_pyabc()
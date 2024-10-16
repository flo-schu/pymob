import pytest
from matplotlib import pyplot as plt
from pymob.solvers import JaxSolver
import numpy as np
import jax

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


def test_model_check():
    sim = init_test_case_study_hierarchical_presimulated("lotka_volterra_hierarchical_presimulated_v1")
    sim = init_simulation_casestudy_api()
    sim.config.inference_numpyro.gaussian_base_distribution = True
    sim.config.jaxsolver.throw_exception = False
    sim.config.jaxsolver.max_steps = 1000

    sim.solver = JaxSolver
    sim.dispatch_constructor()
    sim.set_inferer(backend="numpyro")

    sim.inferer.check_gradients()
    sim.inferer.check_log_likelihood()


def test_vector_field():
    sim = init_test_case_study_hierarchical_presimulated("lotka_volterra_hierarchical_presimulated_v1")
    sim = init_simulation_casestudy_api()
    sim.config.inference_numpyro.gaussian_base_distribution = True
    sim.config.jaxsolver.throw_exception = False
    sim.config.jaxsolver.max_steps = 10_000

    sim.solver = JaxSolver
    sim.dispatch_constructor()
    sim.set_inferer(backend="numpyro")

    sim.config.model_parameters.beta.min = -10
    sim.config.model_parameters.beta.max = 10
    sim.config.model_parameters.alpha.min = -10
    sim.config.model_parameters.alpha.max = 10

    # Define a scalar function of two variables
    def f(theta):
        x = theta["alpha"]
        y = theta["beta"]
        return -((x+2)**2 + (y-2)**2)

    # Compute the gradient function
    gradient_f = jax.grad(f)

    ax = sim.inferer.plot_likelihood_landscape(
        parameters=("alpha", "beta"),
        log_likelihood_func=jax.vmap(f),
        gradient_func=jax.vmap(gradient_f),
    )
    ax.plot(-2,2,ls="", marker="o", color="black")
    ax.figure.savefig(f"{sim.output_path}/test_loglikelihood_gradients.png")

    log_likelihood, grad_log_likelihood = sim.inferer.create_log_likelihood(
        seed=1, return_type="custom", check=False, 
        custom_return_fn=lambda lj, lp, ld: lj,
        vectorize=True,
        gradients=True
    )


    ax = sim.inferer.plot_likelihood_landscape(
        parameters=("beta", "alpha"),
        log_likelihood_func=log_likelihood,
        gradient_func=grad_log_likelihood,
    )

    ax.figure.savefig(f"{sim.output_path}/loglikelihood_gradients.png")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_scripting_api_pyabc()
import pytest
import numpy as np
from click.testing import CliRunner
from matplotlib import pyplot as plt

from tests.fixtures import init_simulation_casestudy_api

def test_diffrax_exception():
    # with proper scripting API define JAX model here or import from fixtures
    sim = init_simulation_casestudy_api("test_scenario")

    # diffrax returns infinity for all computed values after which the solver 
    # breaks due to raching maximum number of steps. 
    # This function calculates the number of inf values
    n_inf = lambda x: (x.results.to_array() == np.inf).values.sum() / len(x.data_variables)

    ub_alpha = 5.0  # alpha values above do not yield reasonable fits for beta = 0.02
    alpha = np.logspace(-2, 3, 20)  # we sample alpha from 0.01-100

    badness = []
    for a in alpha:
        eva = sim.dispatch({"alpha": a, "beta": 0.02})
        eva()

        badness.append(n_inf(eva))

    # the tests make sure that parameters within feasible bounds result in simulation
    # results without inf values and do contain inf values when parameters above
    # feasible bounds are sampled.
    badness_for_feasible_alpha = np.array(badness)[np.where(alpha < ub_alpha)[0]]
    assert sum(badness_for_feasible_alpha) == 0

    badness_for_infeasible_alpha = np.array(badness)[np.where(alpha >= ub_alpha)[0]]
    assert sum(badness_for_infeasible_alpha) > 0


def test_convergence_user_defined_probability_model():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "nuts"
    sim.config.inference_numpyro.user_defined_probability_model = "parameter_only_model"
    sim.config.inference_numpyro.user_defined_preprocessing = "dummy_preprocessing"

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_median = sim.inferer.idata.posterior.median( # type: ignore
        ("chain", "draw"))[["beta", "alpha"]] 
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_median.to_dataarray().values,
        np.array([0.02, 0.5]),
        rtol=1e-1, atol=1e-3
    )


def test_convergence_nuts_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "nuts"
    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )

def test_convergence_svi_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "svi"
    sim.config.inference_numpyro.svi_iterations = 10_000
    sim.config.inference_numpyro.svi_learning_rate = 0.01
    # this samples the model with standard normal distributions
    # and rescales them according to the transformations of the specified 
    # parameter distributions to the normal
    sim.config.inference_numpyro.gaussian_base_distribution = True

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    sim.inferer.idata.posterior_predictive # type: ignore

    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )


    # posterior predictions
    fig, axes = plt.subplots(2,1, sharex=True)
    for data_var, ax in zip(sim.config.data_structure.data_variables, axes):
        ax = sim.inferer.plot_posterior_predictions(
            data_variable=data_var, 
            x_dim="time",
            ax=ax
        )


def test_convergence_map_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "map"
    sim.config.inference_numpyro.svi_iterations = 2000
    sim.config.inference_numpyro.svi_learning_rate = 0.01
    # this samples the model with standard normal distributions
    # and rescales them according to the transformations of the specified 
    # parameter distributions to the normal
    sim.config.inference_numpyro.gaussian_base_distribution = 1

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    sim.inferer.idata.posterior_predictive # type: ignore

    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )



def test_convergence_nuts_kernel_replicated():
    pytest.skip()
    # CURRENTLY UNUSABLE SEE https://github.com/flo-schu/pymob/issues/6
    sim = init_simulation_casestudy_api("test_scenario_replicated")

    sim.config.set("inference.numpyro", "kernel", "nuts")
    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_mean = sim.inferer.idata.posterior.mean(("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )

    

def test_convergence_sa_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "sa"
    sim.config.inference_numpyro.init_strategy = "init_to_sample"
    sim.config.inference_numpyro.warmup = 2000
    sim.config.inference_numpyro.draws = 1000
    sim.config.inference_numpyro.sa_adapt_state_size = 10

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )



    # posterior predictions
    for data_var in sim.config.data_structure.data_variables:
        ax = sim.inferer.plot_posterior_predictions(
            data_variable=data_var, 
            x_dim="time"
        )


def test_commandline_api_infer():
    # TODO: This will run, once methods are available for 
    # - prior_predictive_checks, 
    # - store_results, 
    # - posterior_predictive_checks 
    pytest.skip()
    from pymob.infer import main
    runner = CliRunner()
    
    args = "--case_study=test_case_study "\
        "--scenario=test_scenario "\
        "--inference_backend=numpyro"
    result = runner.invoke(main, args.split(" "))

    if result.exception is not None:
        raise result.exception


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_user_defined_probability_model()
    # test_nuts_kernel()
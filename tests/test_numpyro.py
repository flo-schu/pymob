import numpy as np

from pymob.utils.store_file import prepare_casestudy

def create_simulation(scenario):
    config = prepare_casestudy(
        case_study=("test_case_study", scenario),
        config_file="settings.cfg"
    )
    from case_studies.test_case_study.sim import Simulation
    return Simulation(config=config)

def test_diffrax_exception():
    # with proper scripting API define JAX model here or import from fixtures
    sim = create_simulation("test_scenario")

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


def test_nuts_kernel():
    sim = create_simulation("test_scenario")

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

def test_nuts_kernel_replicated():
    # CURRENTLY UNUSABLE SEE https://github.com/flo-schu/pymob/issues/6
    config = prepare_casestudy(
        case_study=("test_case_study", "test_scenario_replicated"),
        config_file="settings.cfg"
    )
    from case_studies.test_case_study.sim import ReplicatedSimulation
    
    sim = ReplicatedSimulation(config=config)
    sim = create_simulation("test_scenario_replicated")

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

def test_sa_kernel():
    sim = create_simulation("test_scenario")

    sim.config.set("inference.numpyro", "kernel", "sample-adaptive-mcmc")
    sim.config.set("inference.numpyro", "init_strategy", "init_to_sample")
    sim.config.set("inference.numpyro", "warmup", "2000")
    sim.config.set("inference.numpyro", "draws", "1000")
    sim.config.set("inference.numpyro", "sa_adapt_state_size", "10")

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



    # posterior predictions
    for data_var in sim.data_variables:
        ax = sim.inferer.plot_predictions(
            data_variable=data_var, 
            x_dim="time"
        )

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    test_nuts_kernel_replicated()
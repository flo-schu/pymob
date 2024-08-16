import pytest
import numpy as np

from pymob.inference.scipy_backend import ScipyBackend
from pymob.sim.parameters import Param
from pymob.solvers.diffrax import JaxSolver

from tests.fixtures import init_test_case_study_hierarchical

def test_parameter_parsing_different_priors_on_species():
    sim = init_test_case_study_hierarchical()
    
    sim.config.model_parameters.alpha_species = Param(
        value=0.5, free=True, hyper=True,
        dims=('rabbit_species','experiment'),
        # take good care to specify hyperpriors correctly. 
        # Dimensions are broadcasted following the normal rules of 
        # numpy. The below means, in dimension one, we have two different
        # assumptions 1, and 3. Dimension one is the dimension of the rabbit species.
        # The specification loc=[1,3] would be understood as [[1,3]] and
        # be understood as the experiment dimension. Ideally, the dimensionality
        # is so low that you can be specific about the priors. I.e.:
        # scale = [[1,1,1],[3,3,3]]. This of course expects you know about
        # the dimensionality of the prior (i.e. the unique coordinates of the dimensions)
        prior="norm(loc=[[1],[3]],scale=0.1)" # type: ignore
    )
    # prey birth rate
    # to be clear, this says each replicate has a slightly varying birth
    # rate depending on the valley where it was observed. Seems legit.
    sim.config.model_parameters.alpha = Param(
        value=0.5, free=True, hyper=False,
        dims=('id',),
        prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index])" # type: ignore
    )

    # re initialize the observation with
    sim.define_observations_replicated_multi_experiment(n=120) # type: ignore
    sim.coordinates["time"] = np.arange(12)
    y0 = sim.parse_input("y0", drop_dims=["time"])
    sim.model_parameters["y0"] = y0

    inferer = ScipyBackend(simulation=sim)


    theta = inferer.sample_distribution()

    alpha_samples_cottontail = theta["alpha"][sim.observations["rabbit_species"] == "Cottontail"]
    alpha_samples_jackrabbit = theta["alpha"][sim.observations["rabbit_species"] == "Jackrabbit"]

    alpha_cottontail = np.mean(alpha_samples_cottontail)
    alpha_jackrabbit = np.mean(alpha_samples_jackrabbit)
    
    # test if the priors that were broadcasted to the replicates 
    # match the hyperpriors
    np.testing.assert_array_almost_equal(
        [alpha_cottontail, alpha_jackrabbit], [1, 3], decimal=1
    )

    sim.solver = JaxSolver
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
    sim.dispatch_constructor()
    e = sim.dispatch(theta=theta)
    e()

    # res_species = e.results.where(e.results.rabbit_species=="Cottontail", drop=True)
    # store simulated results
    e.results.to_netcdf(
        f"{sim.data_path}/simulated_data_hierarchical_species_year.nc"
    )

    # TODO: mark datavars as observed automatically
    # set observations and mark as observed. This 
    # could be automated by using length of data var > 0 to
    # trigger marking data vars as observed.
    sim.observations = e.results
    sim.config.data_structure.rabbits.observed = True
    sim.config.data_structure.wolves.observed = True

    sim.dispatch_constructor()
    sim.set_inferer("numpyro")

    idata = sim.inferer.prior_predictions()

    # test if numpyro predictions also match the specified priors
    alpha_numpyro = idata.prior["alpha"].mean(("chain", "draw"))
    alpha_numpyro_cottontail = np.mean(alpha_numpyro.values[sim.observations["rabbit_species"] == "Cottontail"])
    alpha_numpyro_jackrabbit = np.mean(alpha_numpyro.values[sim.observations["rabbit_species"] == "Jackrabbit"])
    # test if the priors that were broadcasted to the replicates 
    # match the hyperpriors
    np.testing.assert_array_almost_equal(
        [alpha_numpyro_cottontail, alpha_numpyro_jackrabbit], [1, 3], decimal=1
    )

    try:
        sim.inferer.run()
        raise AssertionError(
            "This model should fail, because there are negative values in the"+
            "observations, hence the log-likelihood becomes nan, because there"+
            "is no support for the values"
        )
    except RuntimeError:
        # check likelihoods of rabbits     
        loglik = sim.inferer.check_log_likelihood(theta)
        nan_liks = np.isnan(loglik[2]["rabbits_obs"]).sum()
        assert nan_liks > 0

    # The conclusion is that this cannot work, because the simulation produces
    # values that are below zero frequently. I need to increase EPS to work
    # sim.config.jaxsolver.throw_exception
    # sim.config.jaxsolver.max_steps = 100000
    # sim.config.inference_numpyro.init_strategy = "init_to_sample"

    # we need to set both observations to values greater zero, because
    # the lognormal distribution has no support for values equal to zero.
    sim.observations = e.results.round() + 1e-6
    if np.any(sim.observations.min().to_array().values < 0):
        raise AssertionError(
            "observations had values < 0. This causes NaN log-likelihoods."
        )


    # we need to set both observations to values greater zero, because
    # the lognormal distribution has no support for values equal to zero.
    # in the inference backend, values returned from the simulator are modified
    sim.config.error_model.rabbits = "lognorm(scale=rabbits+1e-6, s=0.1)"
    sim.config.error_model.rabbits = "lognorm(scale=rabbits+EPS, s=0.1)"
    sim.config.inference.eps = 1e-6
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")

    idata = sim.inferer.prior_predictions()
    idata.prior_predictive.rabbits.min()

    loglik = sim.inferer.check_log_likelihood(theta)
    nan_liks_rabbits = np.isnan(loglik[2]["rabbits_obs"]).sum()
    nan_liks_wolves = np.isnan(loglik[2]["wolves_obs"]).sum()

    np.testing.assert_array_equal([nan_liks_wolves, nan_liks_rabbits], [0,0])


def test_parameter_parsing_different_priors_on_year():
    sim = init_test_case_study_hierarchical()
    
    sim.config.model_parameters.alpha_species = Param(
        value=0.5, free=True, hyper=True, dims=('rabbit_species','experiment'),
        prior="norm(loc=[[1,2,3]],scale=0.1)" # type: ignore
    )
    # prey birth rate
    sim.config.model_parameters.alpha = Param(
        value=0.5, free=True, hyper=True, dims=('id',),
        prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index])" # type: ignore
    )

    # re initialize the observation with
    sim.define_observations_replicated_multi_experiment(n=120) # type: ignore
    y0 = sim.parse_input("y0", drop_dims=["time"])
    sim.model_parameters["y0"] = y0

    inferer = ScipyBackend(simulation=sim)


    theta = inferer.sample_distribution()

    alpha_samples_2010 = theta["alpha"][sim.observations["experiment"] == "2010"]
    alpha_samples_2011 = theta["alpha"][sim.observations["experiment"] == "2011"]
    alpha_samples_2012 = theta["alpha"][sim.observations["experiment"] == "2012"]

    alpha_2010 = np.mean(alpha_samples_2010)
    alpha_2011 = np.mean(alpha_samples_2011)
    alpha_2012 = np.mean(alpha_samples_2012)
    
    # test if the priors that were broadcasted to the replicates 
    # match the hyperpriors
    np.testing.assert_array_almost_equal(
        [alpha_2010, alpha_2011, alpha_2012], [1, 2, 3], decimal=1
    )




if __name__ == "__main__":
    pass
    # test_parameter_parsing_different_priors_on_species()
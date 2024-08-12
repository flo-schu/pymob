import pytest
import numpy as np
from pymob.inference.scipy_backend import ScipyBackend
from pymob.sim.parameters import Param
from tests.fixtures import init_test_case_study_hierarchical

def test_parameter_parsing_different_priors_on_species():
    sim = init_test_case_study_hierarchical()
    
    sim.config.model_parameters.alpha_species = Param(
        value=0.5, free=True, hyper=True,
        # take good care to specify hyperpriors correctly. 
        # Dimensions are broadcasted following the normal rules of 
        # numpy. The below means, in dimension one, we have two different
        # assumptions 1, and 3. Dimension one is the dimension of the rabbit species.
        # The specification loc=[1,3] would be understood as [[1,3]] and
        # be understood as the experiment dimension. Ideally, the dimensionality
        # is so low that you can be specific about the priors. I.e.:
        # scale = [[1,1,1],[3,3,3]]. This of course expects you know about
        # the dimensionality of the prior (i.e. the unique coordinates of the dimensions)
        prior="norm(loc=[[1],[3]],scale=0.1,dims=('rabbit_species','experiment'))" # type: ignore
    )
    # prey birth rate
    sim.config.model_parameters.alpha = Param(
        value=0.5, free=True, hyper=True,
        prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index],dims=('id',))" # type: ignore
    )

    # re initialize the observation with
    sim.define_observations_replicated_multi_experiment(n=120) # type: ignore
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


def test_parameter_parsing_different_priors_on_year():
    sim = init_test_case_study_hierarchical()
    
    sim.config.model_parameters.alpha_species = Param(
        value=0.5, free=True, hyper=True,
        prior="norm(loc=[[1,2,3]],scale=0.1,dims=('rabbit_species','experiment'))" # type: ignore
    )
    # prey birth rate
    sim.config.model_parameters.alpha = Param(
        value=0.5, free=True, hyper=True,
        prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index],dims=('id',))" # type: ignore
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
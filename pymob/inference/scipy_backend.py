from functools import partial
from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import Generator, PCG64
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete, rv_generic, rv_continuous_frozen, rv_discrete_frozen

from pymob.sim.parameters import scipy_to_scipy
from pymob.inference.base import Distribution, Errorfunction, InferenceBackend
from pymob.simulation import SimulationBase
from pymob.utils.config import lookup
from pymob.inference.error_models import ErrorModel

class ScipyDistribution(Distribution):
    distribution_map: Dict[str,Tuple[Union[rv_continuous,rv_discrete],Dict[str,str]]] = scipy_to_scipy
    parameter_converter = staticmethod(lambda x: np.array(x))
    
    @property
    def dist_name(self) -> str:
        return self.distribution.name
    

    
class ScipyBackend(InferenceBackend):
    _distribution = ScipyDistribution
    distribution: Union[rv_continuous,rv_discrete]

    def __init__(self, simulation: SimulationBase) -> None:
        super().__init__(simulation)
        self.inference_model = self.parse_probabilistic_model()
        self.random_state = Generator(PCG64(self.config.simulation.seed))

    def parse_deterministic_model(self):
        pass

    def parse_probabilistic_model(self):
        return ProbabilisticModel(
            prior_model=self.prior,
            error_model=self.error_model,
            indices=self.indices,
            observations=self.simulation.observations,
            simulation=self.simulation,
            eps=self.config.inference.eps,
            seed=self.config.simulation.seed,
        )


    def posterior_predictions(self):
        pass

    def prior_predictions(self):
        return self.inference_model(theta=None, observations=None)


    def sample_distribution(self):
        return self.inference_model.prior_model()
    
    def create_log_likelihood(self) -> Tuple[Errorfunction,Errorfunction]:
        # TODO: define
        return 


class ScipyPriorModel:
    def __init__(self, prior_model, indices, observations, seed):
        self.prior_model = prior_model
        self.random_state = Generator(PCG64(seed))
        
        self.context = [
            indices, observations
        ]

    def __call__(self, theta=None):
        if theta is None:
            return self.forward()
        else:
            return self.reverse(theta=theta)

    def _draw_random_variables(self) -> Dict[str, rv_generic]:
        prior_samples = {}

        # prior is added here, so it is updated 
        context = [prior_samples] + self.context
        for name, prior in self.prior_model.items():

            dist = prior.construct(context=context)


            sample = dist.rvs(size=prior.shape, random_state=self.random_state)

            prior_samples.update({name:sample})

        return prior_samples


    def _calc_log_prob(self, theta) -> Dict[str, rv_generic]:
        prior_samples = {}

        # prior is added here, so it is updated 
        context = [prior_samples] + self.context
        for name, prior in self.prior_model.items():

            dist = prior.construct(context=context)

            if hasattr(dist, "logpdf"):
                logprob = dist.logpdf(theta[name])
            elif hasattr(dist, "logpmf"):
                logprob = dist.logpmf(theta[name])
            else:
                raise NotImplementedError(
                    "scipy distribution must by rv_continuous or rv_discrete"
                )

            prior_samples.update({name: logprob})

        return prior_samples

    def forward(self):
        return self._draw_random_variables()
    
    def reverse(self, theta):
        return self._calc_log_prob(theta=theta)


class ScipyErrorModel(ErrorModel):
    def __init__(self, eps, error_model, indices, observations, seed):
        extra = {"EPS": eps, "np": np}
        self.error_model = error_model
        self.random_state = Generator(PCG64(seed))

        self.context = [
            indices,
            observations,
            extra
        ]


    def _parameterize_random_variables(self, Y) -> Dict[str, rv_generic]:
        """Parameterizes random variables from Expression-based error models
        """
        distributions = {}
        for error_model_name, error_model_dist in self.error_model.items():
            rv = error_model_dist.construct(
                context=[Y] + self.context
            )

            distributions.update({error_model_name: rv})
                                
        return distributions


    def forward(self, Y):
        random_variables = self._parameterize_random_variables(Y=Y)
        return {
            key: rv.rvs(random_state=self.random_state) 
            for key, rv in random_variables.items()
        }

    def reverse(self, Y, Y_obs):
        random_variables = self._parameterize_random_variables(Y=Y)

        likelihoods = {}
        for key, rv in random_variables.items():
            if hasattr(rv, "logpdf"):
                logprob = rv.logpdf(Y_obs[key])
            elif hasattr(rv, "logpmf"):
                logprob = rv.logpmf(Y_obs[key])
            else:
                raise NotImplementedError(
                    "scipy distribution must by rv_continuous or rv_discrete"
                )
            
            likelihoods.update({key: logprob})

        return likelihoods
    

class ScipyTransModel:
    def __init__(self, simulation):
        self.simulation = simulation

    def transform_prior_to_error_model(self, theta, y0={}, x_in={}, seed=None):
        evaluator = self.simulation.dispatch(theta=theta, y0=y0, x_in=x_in)
        evaluator(seed)
        return evaluator.Y
    
    def __call__(self, theta, y0={}, x_in={}, seed=None):
        return self.transform_prior_to_error_model(theta, y0, x_in, seed)


class ProbabilisticModel:

    def __init__(
        self,
        prior_model,
        error_model,
        indices,
        observations,
        simulation,
        eps,
        seed
    ):
        rng = np.random.default_rng(seed=seed)
        # split the seed into two additional random seeds
        seeds = rng.integers(low=1, high=1000, size=2)

        self.prior_model = ScipyPriorModel(
            prior_model=prior_model, indices=indices, observations=observations, seed=seeds[0]
        )
        self.trans_model = ScipyTransModel(
            simulation=simulation
        )
        self.error_model = ScipyErrorModel(
            eps=eps, error_model=error_model, indices=indices, observations=observations, seed=seeds[1]
        )

    def __call__(self, theta=None, observations=None):
        """Evaluate the inference model in various modes and calculate likelihoods

        prior predictions
        -----------------
        - theta = None
        - observations = None 

        When no paramaters and observations are specified, this corresponds to prior predictions
        parameters samples and noisy observations are generated
        
        
        likelihood
        ----------
        - theta defined
        - observations defined

        No random draws are made, but the probabilites of the passed parameters and 
        observations are computed

        
        sampling
        --------
        - theta = None
        - observations defined

        This reduces to a rudimentary markov sampler that generates random draws from
        the parameter distributions and evaluates the given observations w.r.t to
        the error model

        
        posterior_predictive
        --------------------
        - theta defined
        - observations = None

        When theta is defined and observations are undefined, random draws of the observations
        are generated, based on the error distributions

        
        """
        if theta is None:
            theta = self.prior_model()

        
        theta_prob = self.prior_model(theta)
        results = self.trans_model(theta)

        if observations is None:
            observations = self.error_model(Y=results)
        
        error_prob = self.error_model(Y=results, Y_obs=observations)

        return {
            "theta": theta, 
            "theta_prob": theta_prob, 
            "results": results,
            "observations": observations, 
            "observations_prob": error_prob
        }
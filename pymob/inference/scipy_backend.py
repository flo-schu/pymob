from functools import partial
from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import Generator, PCG64
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete, rv_generic
from scipy.stats._multivariate import multi_rv_generic

from pymob.sim.config import scipy_to_scipy
from pymob.inference.base import Distribution, Errorfunction, InferenceBackend
from pymob.simulation import SimulationBase
from pymob.utils.config import lookup
from pymob.inference.error_models import ErrorModel

class ScipyDistribution(Distribution):
    """Distribution wrapper for SciPy random variables.

    This subclass of :class:`~pymob.inference.base.Distribution` provides a thin
    wrapper around SciPy's continuous and discrete distributions. It maps a
    distribution name to the corresponding SciPy ``rv_continuous`` or
    ``rv_discrete`` object using the ``scipy_to_scipy`` dictionary. The class
    also defines a ``parameter_converter`` that converts parameter arrays to
    NumPy ``ndarray`` objects.

    The primary purpose of this class is to expose a ``dist_name`` property that
    returns the name of the underlying SciPy distribution.
    """
    distribution_map: Dict[str,Tuple[Union[rv_continuous,rv_discrete,multi_rv_generic],Dict[str,str]]] = scipy_to_scipy
    parameter_converter = staticmethod(lambda x: np.array(x))
    
    @property
    def dist_name(self) -> str:
        return self.distribution.name
    

    
class ScipyBackend(InferenceBackend):
    """Backend that uses SciPy distributions for inference.

    The backend implements the abstract :class:`~pymob.inference.base.InferenceBackend`
    interface using SciPy probability distributions. It parses the model priors
    and error models from the simulation configuration, builds a
    :class:`ProbabilisticModel` that can generate prior predictive samples,
    compute likelihoods, and sample from the prior distribution.

    Attributes
    ----------
    inference_model : ProbabilisticModel
        The assembled probabilistic model used for all inference operations.
    random_state : numpy.random.Generator
        Random number generator seeded from the simulation configuration.
    """
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
    
    def create_log_likelihood(self) -> Tuple[Errorfunction, Errorfunction]:
        """Create log-likelihood and (optional) gradient functions.

        Returns
        -------
        tuple
            A pair ``(log_likelihood, gradient)`` where each element is a
            callable conforming to the :class:`Errorfunction` protocol.
            The current backend does not implement a gradient, so both callables
            simply return ``None``. This stub satisfies the type checker and can be
            extended in the future.
        """
        # Simple placeholder implementations
        def _log_likelihood(theta):
            # Placeholder: actual implementation not provided
            return None

        def _gradient(theta):
            # Placeholder: gradient not implemented
            return None

        return _log_likelihood, _gradient
    
    def run(self):
        pass


class ScipyPriorModel:
    """Helper class for sampling from prior distributions and computing log-probabilities.

    Instances maintain a reference to the prior model definition, the index
    variables, and observations. The ``__call__`` method forwards to either
    ``forward`` (when no parameters are supplied) or ``reverse`` (when a
    parameter dictionary is provided), enabling a uniform callable interface.

    Parameters
    ----------
    prior_model : dict
        Mapping of parameter names to :class:`Distribution` objects.
    indices : dict
        Index arrays for each indexed dimension of the simulation.
    observations : xarray.Dataset
        Observed data used for conditioning (currently not used in the prior).
    seed : int
        Seed for the underlying NumPy random generator.
    """
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
    """Error model that generates observation noise using SciPy distributions.

    The class builds a set of random variables based on the user-specified error
    model expressions. It can draw synthetic noisy observations from the model
    (``forward``) or compute the log-probability of observed data given a set of
    latent variables (``reverse``).

    Parameters
    ----------
    eps : float
        Small constant added to scales to avoid division by zero.
    error_model : dict
        Mapping of data variable names to error model :class:`Distribution` objects.
    indices : dict
        Index arrays for the simulation.
    observations : xarray.Dataset
        Observed data used for likelihood evaluation.
    seed : int
        Seed for the random number generator.
    """
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
    """Transformation model that runs the deterministic simulation.

    This lightweight wrapper forwards the ``theta`` (parameter values) to the
    simulation's ``dispatch`` method, executes the model, and returns the
    resulting simulated state ``Y``. It is used by the probabilistic model to
    generate latent system states from sampled parameters.
    """
    def __init__(self, simulation):
        self.simulation = simulation

    def transform_prior_to_error_model(self, theta, y0={}, x_in={}, seed=None):
        evaluator = self.simulation.dispatch(theta=theta, y0=y0, x_in=x_in)
        evaluator(seed)
        return evaluator.Y
    
    def __call__(self, theta, y0={}, x_in={}, seed=None):
        return self.transform_prior_to_error_model(theta, y0, x_in, seed)


class ProbabilisticModel:
    """Combined prior, transformation, and error model for inference.

    The class orchestrates three components:

    * **Prior model** - draws parameter samples and evaluates their log-probability.
    * **Transformation model** - runs the deterministic simulation to obtain latent
      system states.
    * **Error model** - generates synthetic observations or evaluates the likelihood
      of observed data.

    The ``__call__`` method implements four usage modes (prior predictive,
    likelihood, sampling, posterior predictive) as described in the docstring of
    ``ScipyBackend``.
    """

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

from functools import partial, lru_cache
import glob
import re
import warnings
from typing import Tuple, Dict, Union, Optional, Callable
from tqdm import tqdm
import numpyro
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist
from numpyro.infer import Predictive
from numpyro.distributions import Normal, transforms, TransformedDistribution
from numpyro import infer
import xarray as xr
import arviz as az
from matplotlib import pyplot as plt
import sympy
import sympy2jax

from pymob.simulation import SimulationBase
from pymob.inference.analysis import rename_extra_dims

sympy2jax_extra_funcs = {
    sympy.Array: jnp.array,
    sympy.Tuple: tuple,
}

def LogNormalTrans(loc, scale):
    return TransformedDistribution(
        base_distribution=Normal(0,1), 
        transforms=[
            transforms.AffineTransform(loc=jnp.log(loc), scale=scale), 
            exp()
        ])

def HalfNormalTrans(scale):
    return TransformedDistribution(
        base_distribution=Normal(0,1), 
        transforms=[
            transforms.AffineTransform(loc=0, scale=scale), 
            transforms.AbsTransform()
        ])


def transform(transforms, x):
    for part in transforms:
        x = part(x)
    return x

def inv_transform(transforms, y):
    for part in transforms[::-1]:
        y = part.inv(y)
    return y


def catch_patterns(expression_str):
    # tries to match array notation [0 1 2]
    pattern = r"\[(\d+(\.\d+)?(\s+\d+(\.\d+)?)*|\s*)\]"
    if re.fullmatch(pattern, expression_str) is not None:
        expression_str = expression_str.replace(" ", ",") \
            .removeprefix("[").removesuffix("]")
        return f"stack({expression_str})"

    return expression_str

def generate_transform(expression_str):
    # check for parentheses in expression
    
    expression_str = catch_patterns(expression_str)

    # Parse the expression without knowing the symbol names in advance
    parsed_expression = sympy.sympify(expression_str, evaluate=False)
    free_symbols = tuple(parsed_expression.free_symbols)

    # Transform expression to jax expression
    func = sympy2jax.SymbolicModule(
        parsed_expression, 
        extra_funcs=sympy2jax_extra_funcs, 
        make_array=True
    )

    return {"transform": func, "args": [str(s) for s in free_symbols]}

exp = transforms.ExpTransform
sigmoid = transforms.SigmoidTransform
C = transforms.ComposeTransform

class NumpyroBackend:
    def __init__(
        self, 
        simulation: SimulationBase
    ):
        """Initializes the NumpyroBackend with a Simulation object.

        Parameters
        ----------
        simulation : SimulationBase
            An initialized simulation.
        """
        self.EPS = 1e-8
        self.distribution_map = {
            "lognorm": (LogNormalTrans, {"scale": "loc", "s": "scale"}),
            "binom": (dist.Binomial, {"n":"total_count", "p":"probs"}),
            "normal": dist.Normal,
            "halfnorm": HalfNormalTrans,
            "poisson": (dist.Poisson, {"mu": "rate"}),
        }
        
        self.simulation = simulation
        self.evaluator = self.parse_deterministic_model()
        self.prior = self.parse_model_priors()
        self.error_model = self.parse_error_model()

        # parse preprocessing
        if self.user_defined_preprocessing is not None:
            self.preprocessing = getattr(
                self.simulation.prob,
                self.user_defined_preprocessing
            )

        # parse the probability model
        if self.user_defined_probability_model is not None:
            self.inference_model = getattr(
                self.simulation.prob, 
                self.user_defined_probability_model
            )
        elif self.gaussian_base_distribution:
            self.inference_model = self.parse_probabilistic_model_with_gaussian_base()
        else:
            self.inference_model = self.parse_probabilistic_model()


        self.idata: az.InferenceData

    @property
    def plot_function(self) -> Optional[str]:
        return self.simulation.config.get(
            "inference", "plot_function", 
            fallback=None
        )

    @property
    def extra_vars(self):
        extra = self.simulation.config.getlist( 
            "inference", "extra_vars", fallback=[]
        )
        return extra if isinstance(extra, list) else [extra]
    

    @property
    def user_defined_probability_model(self):
        return self.simulation.config.get(
            "inference.numpyro", "user_defined_probability_model", fallback=None
        )
    
    @property
    def user_defined_preprocessing(self):
        return self.simulation.config.get(
            "inference.numpyro", "user_defined_preprocessing", fallback=None
        )

    @property
    def gaussian_base_distribution(self):
        flag = self.simulation.config.getint(
            "inference.numpyro", "gaussian_base_distribution", fallback=False
        )
        return bool(flag)
    
    @property
    def n_predictions(self):
        return self.simulation.config.getint(
            "inference", "n_predictions", fallback=100
        )
    

    @property
    def chains(self):
        return self.simulation.config.getint(
            "inference.numpyro", "chains", fallback=1
        )
    
    @property
    def draws(self):
        return self.simulation.config.getint(
            "inference.numpyro", "draws", fallback=1000
        )
    
    @property
    def warmup(self):
        return self.simulation.config.getint(
            "inference.numpyro", "warmup", fallback=self.draws
        )
    
    @property
    def thinning(self):
        return self.simulation.config.getint(
            "inference.numpyro", "thinning", fallback=1
        )

    @property
    def svi_iterations(self):
        return self.simulation.config.getint(
            "inference.numpyro", "svi_iterations", fallback=10_000
        )
    
    @property
    def svi_learning_rate(self):
        return self.simulation.config.getfloat(
            "inference.numpyro", "svi_learning_rate", fallback=0.0001
        )
    
    @property
    def kernel(self):
        return self.simulation.config.get(
            "inference.numpyro", "kernel", fallback="nuts"
        )
    

    @property
    def adapt_state_size(self):
        return self.simulation.config.getint(
            "inference.numpyro", "sa_adapt_state_size", fallback=None
        )

    @property
    def init_strategy(self):
        strategy = self.simulation.config.get(
            "inference.numpyro", "init_strategy", fallback="init_to_median"
        )

        return getattr(infer, strategy)

    def parse_deterministic_model(self) -> Callable:
        """Parses an evaluation function from the Simulation object, which 
        takes a single argument theta and defaults to passing no seed to the
        deterministic evaluator.

        Returns
        -------
        callable
            The evaluation function
        """
        def evaluator(theta, seed=None):
            evaluator = self.simulation.dispatch(theta=theta)
            evaluator(seed)
            return evaluator.Y
        
        return evaluator

    def model(self):
        pass


    def observation_parser(self) -> Tuple[Dict,Dict]:
        """Transform a xarray.Dataset into a dictionary of jnp.Arrays. Creates
        boolean arrays of masks for nan values (missing values are tagged False)

        Returns
        -------
        Tuple[Dict,Dict]
            Dictionaries of observations (data) and masks (missing values)
        """
        obs = self.simulation.observations \
            .transpose(*self.simulation.dimensions)
        data_vars = self.simulation.data_variables + self.extra_vars

        masks = {}
        observations = {}
        for d in data_vars:
            o = jnp.array(obs[d].values)
            m = jnp.logical_not(jnp.isnan(o))
            observations.update({d:o})
            masks.update({d:m})
        
        return observations, masks
    
    @staticmethod
    def parse_parameter(parname: str, prior: str, distribution_map: Dict[str,Tuple]) -> Tuple[str,dist.Distribution,Dict[str,Callable]]:

        distribution, cluttered_arguments = prior.split("(", 1)
        param_strings = cluttered_arguments.split(")", 1)[0].split(",")

        distribution_mapping = distribution_map[distribution]

        if not isinstance(distribution_mapping, tuple):
            distribution = distribution_mapping
            distribution_mapping = (distribution, {})
        
        assert len(distribution_mapping) == 2, (
            "distribution and parameter mapping must be "
            "a tuple of length 2."
        )

        distribution, parameter_mapping = distribution_mapping
        mapped_params = {}
        for parstr in param_strings:
            key, val = parstr.split("=")
            
            mapped_key = parameter_mapping.get(key, key)

            # if is_number(val):
            #     parsed_val = float(val)
            # else:
            parsed_val = generate_transform(expression_str=val)

            # parsed_val = float(val) if is_number(val) else val
            mapped_params.update({mapped_key:parsed_val})

        return parname, distribution, mapped_params

    def parse_model_priors(self):
        priors = {}
        for par in self.simulation.free_model_parameters:
            name, distribution, params = self.parse_parameter(
                parname=par.name, 
                prior=par.prior, 
                distribution_map=self.distribution_map
            )
            # parameterized_dist = distribution(**params)
            priors.update({
                name: {
                    "fn":distribution, 
                    "parameters": params
                }
            })
        return priors

    def parse_error_model(self):
        error_model = {}
        for data_var, error_distribution in self.simulation.error_model.items():
            name, distribution, parameters = self.parse_parameter(
                parname=data_var,
                prior=error_distribution,
                distribution_map=self.distribution_map
            )

                
            error_model.update({
                data_var: {
                    "fn": distribution, 
                    "parameters": parameters,
                    "error_dist": error_distribution
                }
            })
        return error_model

    def parse_probabilistic_model(self):
        EPS = self.EPS
        prior = self.prior.copy()
        error_model = self.error_model.copy()
        extra = {"EPS": EPS}

        def lookup(val, deterministic, prior_samples, observations):
            if val in deterministic:
                return deterministic[val]
            
            elif val in prior_samples:
                return prior_samples[val]
            
            elif val in observations:
                return observations[val]
            
            elif val in extra:
                return extra[val]

            else:
                return val

        def model(solver, obs, masks):
            # construct priors with numpyro.sample and sample during inference
            theta = {}
            for prior_name, prior_kwargs in prior.items():
                
                # apply transforms to priors. This will be handy when I use nested
                # parameters
                prior_distribution_parameters = {}
                for pri_par, pri_val in prior_kwargs["parameters"].items():
                    prior_trans_func = pri_val["transform"]
                    prior_trans_func_kwargs = {k: lookup(k, {}, theta, obs) for k in pri_val["args"]}
                    prior_distribution_parameters.update({
                        pri_par: prior_trans_func(**prior_trans_func_kwargs)
                    })

                theta_i = numpyro.sample(
                    name=prior_name,
                    fn=prior_kwargs["fn"](**prior_distribution_parameters)
                )

                theta.update({prior_name: theta_i})
            
            # calculate deterministic simulation with parameter samples
            sim_results = solver(theta=theta)

            # store data_variables as deterministic model output
            for deterministic_name, deterministic_value in sim_results.items():
                _ = numpyro.deterministic(
                    name=deterministic_name, 
                    value=deterministic_value
                )

            for error_model_name, error_model_kwargs in error_model.items():
                error_distribution = error_model_kwargs["fn"]
                error_distribution_kwargs = {}
                for errdist_par, errdist_val in error_model_kwargs["parameters"].items():
                    errdist_trans_func = errdist_val["transform"]
                    errdist_trans_func_kwargs = {
                        k: lookup(k, sim_results, theta, obs) for k in errdist_val["args"]
                    }
                    error_distribution_kwargs.update({
                        errdist_par: errdist_trans_func(**errdist_trans_func_kwargs)
                    })

                _ = numpyro.sample(
                    name=error_model_name + "_obs",
                    fn=error_distribution(**error_distribution_kwargs).mask(masks[error_model_name]),
                    obs=obs[error_model_name]
                )


            # TODO: How to add EPS
            # DONE: add biomial n --> I think this is already done
            # numpyro.sample("cext_obs", LogNormalTrans(loc=cext + EPS, scale=sigma_cext).mask(masks["cext"]), obs=obs["cext"] + EPS)
            # numpyro.sample("cint_obs", LogNormalTrans(loc=cint + EPS, scale=sigma_cint).mask(masks["cint"]), obs=obs["cint"] + EPS)
            # numpyro.sample("nrf2_obs", LogNormalTrans(loc=nrf2 + EPS, scale=sigma_nrf2).mask(masks["nrf2"]), obs=obs["nrf2"] + EPS)
            # numpyro.sample("lethality_obs", dist.Binomial(probs=leth, total_count=obs["nzfe"]).mask(masks["lethality"]), obs=obs["lethality"])
        return model

    def parse_probabilistic_model_with_gaussian_base(self):
        EPS = self.EPS
        prior = self.prior.copy()
        error_model = self.error_model.copy()
        extra = {"EPS": EPS}


        def lookup(val, deterministic, prior_samples, observations):
            if val in deterministic:
                return deterministic[val]
            
            elif val in prior_samples:
                return prior_samples[val]
            
            elif val in observations:
                return observations[val]
            
            elif val in extra:
                return extra[val]

            else:
                return val

        def model(solver, obs, masks):
            theta = {}
            theta_base = {}
            for prior_name, prior_kwargs in prior.items():

                # apply transforms to priors. This will be handy when I use nested
                # parameters
                prior_distribution_parameters = {}
                for pri_par, pri_val in prior_kwargs["parameters"].items():
                    prior_trans_func = pri_val["transform"]
                    # TODO could be replaced by utils.config.lookup and lookup args
                    prior_trans_func_kwargs = {k: lookup(k, {}, theta, obs) for k in pri_val["args"]}
                    prior_distribution_parameters.update({
                        pri_par: prior_trans_func(**prior_trans_func_kwargs)
                    })

                # parameterize the prior distribution and extract transforms
                dist = prior_kwargs["fn"](**prior_distribution_parameters)
                
                try:
                    transforms = getattr(dist, "transforms")
                except:
                    raise RuntimeError(
                        "The specified distribution had no transforms. If setting "
                        "the option 'inference.numpyro.gaussian_base_distribution = 1', "
                        "you are only allowed to use parameter distribution, which can "
                        "be specified as transformed normal distributions. "
                        "Currently only 'lognorm' and 'halfnorm' are implemented. "
                        "You can use the numypro.distributions.TransformedDistribution "
                        "API to specify additional distributions with transforms."
                        "And pass them to the inferer by updating the distribution map: "
                        "sim.inferer.distribution_map.update({'newdist': your_new_distribution})"
                    )

                # sample from a random normal distribution
                theta_base_i = numpyro.sample(
                    name=f"{prior_name}_normal_base",
                    fn=Normal(loc=0, scale=1),
                    sample_shape=dist.shape()                    
                )

                # apply the transforms 
                theta_i = numpyro.deterministic(
                    name=prior_name,
                    value=transform(transforms=transforms, x=theta_base_i)
                )

                theta_base.update({prior_name: theta_base_i})
                theta.update({prior_name: theta_i})

            # calculate deterministic simulation with parameter samples
            sim_results = solver(theta=theta)

            # store data_variables as deterministic model output
            for deterministic_name, deterministic_value in sim_results.items():
                _ = numpyro.deterministic(
                    name=deterministic_name, 
                    value=deterministic_value
                )

            for error_model_name, error_model_kwargs in error_model.items():
                error_distribution = error_model_kwargs["fn"]
                error_distribution_kwargs = {}
                for errdist_par, errdist_val in error_model_kwargs["parameters"].items():
                    errdist_trans_func = errdist_val["transform"]
                    errdist_trans_func_kwargs = {
                        k: lookup(k, sim_results, theta, obs) for k in errdist_val["args"]
                    }
                    error_distribution_kwargs.update({
                        errdist_par: errdist_trans_func(**errdist_trans_func_kwargs)
                    })

                _ = numpyro.sample(
                    name=error_model_name + "_obs",
                    fn=error_distribution(**error_distribution_kwargs).mask(masks[error_model_name]),
                    obs=obs[error_model_name]
                )


            # TODO: How to add EPS
            # DONE: add biomial n --> I think this is already done
            # numpyro.sample("cext_obs", LogNormalTrans(loc=cext + EPS, scale=sigma_cext).mask(masks["cext"]), obs=obs["cext"] + EPS)
            # numpyro.sample("cint_obs", LogNormalTrans(loc=cint + EPS, scale=sigma_cint).mask(masks["cint"]), obs=obs["cint"] + EPS)
            # numpyro.sample("nrf2_obs", LogNormalTrans(loc=nrf2 + EPS, scale=sigma_nrf2).mask(masks["nrf2"]), obs=obs["nrf2"] + EPS)
            # numpyro.sample("lethality_obs", dist.Binomial(probs=leth, total_count=obs["nzfe"]).mask(masks["lethality"]), obs=obs["lethality"])
        return model


    @staticmethod
    def preprocessing(**kwargs):
        return kwargs

    def run(self, print_debug=True):
        # set parameters of JAX and numpyro
        # jax.config.update("jax_enable_x64", True)

        # generate random keys
        key = jax.random.PRNGKey(self.simulation.seed)
        key, *subkeys = jax.random.split(key, 20)
        keys = iter(subkeys)

        # parse observations and masks for missing data
        obs, masks = self.observation_parser()

        model_kwargs = self.preprocessing(
            obs=obs, 
            masks=masks,
        )

        # prepare model and print information about shapes
        model = partial(
            self.inference_model, 
            solver=self.evaluator, 
            **model_kwargs
        )    

        graph = numpyro.render_model(model)
        graph.render(
            filename=f"{self.simulation.output_path}/probability_model",
            view=False, cleanup=True, format="png"
        ) 

        if print_debug:
            with numpyro.handlers.seed(rng_seed=1):
                trace = numpyro.handlers.trace(model).get_trace()
            print(numpyro.util.format_shapes(trace))
            
        if self.kernel.lower() == "sample-adaptive-mcmc":
            kernel = infer.SA(
                model=model,
                dense_mass=True,
                adapt_state_size=self.adapt_state_size,
                init_strategy=self.init_strategy,
            )

        elif self.kernel.lower() == "nuts":
            kernel = infer.NUTS(
                model, 
                dense_mass=True, 
                step_size=0.01,
                adapt_mass_matrix=True,
                adapt_step_size=True,
                max_tree_depth=10,
                target_accept_prob=0.8,
                init_strategy=self.init_strategy
            )

        elif self.kernel.lower() == "svi":
            # The current implementation of SVI only works with parameter distributions
            # that can be derived from normal distributions.
            # what SVI

            if not self.gaussian_base_distribution:
                raise RuntimeError(
                    "SVI is only supported if parameter distributions can be "
                    "re-parameterized as gaussians. Please set "
                    "inference.numpyro.gaussian_base_distribution = 1 "
                    "and if needed use distributions from the loc-scale family "
                    "to specify the model parameters."
                )

            guide = infer.autoguide.AutoMultivariateNormal(model)
            optimizer = numpyro.optim.ClippedAdam(step_size=self.svi_learning_rate, clip_norm=10)
            svi = infer.SVI(model=model, guide=guide, optim=optimizer, loss=infer.Trace_ELBO())
            svi_result = svi.run(next(keys), self.svi_iterations)

            self.idata = self.svi_posterior(
                svi_result, model, guide, next(keys), self.draws
            )
            az.summary(self.idata)

            # the full cov matrix
            cov = svi_result.params['auto_scale_tril'].dot(
                svi_result.params['auto_scale_tril'].T
            )
            median = guide.median(svi_result.params)
            return
        
        elif self.kernel.lower() == "map":
            # while this API is very convenient. I would prefer to roll out my
            # own inference scheme and use scipy's optimizers to minimize 
            # the negative log likelihood for maximum likelihood and to minimize
            # the log-prior and log-likelihood for the map estimate.
            # for initializing this, see the log_density function from above
            # but before i do something like this. Remember, that I then need
            # to deal with passing the right parameter structure to the function
            # so maybe this is something for pymc
            # TODO: It may be worth investing a little bit of extra work in
            # constraining the optimizer
            guide = numpyro.infer.autoguide.AutoDelta(model)
            optimizer = numpyro.optim.ClippedAdam(step_size=self.svi_learning_rate, clip_norm=10)
            svi = infer.SVI(model=model, guide=guide, optim=optimizer, loss=infer.Trace_ELBO())
            svi_result = svi.run(next(keys), num_steps=self.svi_iterations)
            self.idata = self.svi_posterior(
                svi_result, model, guide, next(keys), n=1
            )
            az.summary(self.idata)
            return

        mcmc = infer.MCMC(
            sampler=kernel,
            num_warmup=self.warmup,
            num_samples=self.draws * self.thinning,
            num_chains=self.chains,
            thinning=self.thinning,
            progress_bar=True,
        )
    
        # run inference
        mcmc.run(next(keys))
        mcmc.print_summary()

        # create arviz idata
        idata = az.from_numpyro(
            mcmc, 
            dims=self.posterior_data_structure,
            coords=self.posterior_coordinates,
        )
        self.idata = idata

    @property
    def posterior(self):
        return self.idata.posterior


    @property
    def posterior_data_structure(self):
        data_structure = self.simulation.data_structure.copy()
        data_structure_loglik = {f"{dv}_obs": dims for dv, dims in data_structure.items()}
        data_structure.update(data_structure_loglik)
        return data_structure
    
    @property
    def posterior_coordinates(self):
        posterior_coords = self.simulation.coordinates.copy()
        posterior_coords.update({
            "draw": list(range(self.draws)), 
            "chain": list(range(self.chains))
        })
        return posterior_coords
    

    def create_log_likelihood(self, seed=1):
        key = jax.random.PRNGKey(seed)
        obs, masks = self.observation_parser()

        model_kwargs = self.preprocessing(
            obs=obs, 
            masks=masks,
        )
        
        # prepare model
        model = partial(
            self.inference_model, 
            solver=self.evaluator, 
            **model_kwargs
        )    

        seeded_model = numpyro.handlers.seed(model, key)
   
        def log_density(theta, return_type="summed-by-site", check=True):
            """Log density relies heavily on the substitute utility
            
            The log density is synonymous for log-likelihood (it is the log 
            probability density of the model). 

            The general method is actually quite simple. Values of all SAMPLE
            sites are replaced according to the key: value pairs in `theta`.

            Then the model is calculated and the trace is obtained. Everything
            else is then just post-processing of the sites. Here the log_prob
            function of the sites in the trace are used and the values of the
            sites are inserted. 

            Note that the log-density can randomly fluctuate, if not all
            sites are replaced.

            Note that the data-loglik can be used to calculate a maximum-likelihood
            estimate. Because it is independent of the prior

            Parameters
            ----------

            theta : Dict
                Dictionary of priors (sites) which should be deterministically 
                fixed (substituted).

            return_type : str
                The information which should be returned. With increasing level
                of computation:
                
                joint-log-likelihood: returns a single value, the entire log
                    likelihood of the model, given the values in theta
                full: joint-log, loglik-prior of each site and value, 
                    loglik-data of each site and value 
                summed-by-site: joint-loglik, loglik-prior of sites, 
                    loglik-data of sites
                summed-by-prior-data:
                    joint-loglik, prior-loglik, data-loglik

            """
            
            # TODO: For speeding this up, get inspired by
            # https://num.pyro.ai/en/stable/_modules/numpyro/infer/util.html#log_likelihood

        
            joint_log_density, trace = numpyro.infer.util.log_density(
                model=seeded_model,
                model_args=(),
                model_kwargs={},
                params=theta
            )

            joint_log_density = float(joint_log_density)

            if check:
                sites_in_theta = {
                    name: True 
                    if name in theta
                    else False
                    for name, site in {
                        name: site for name, site in trace.items() 
                        if site["type"] == "sample" 
                        and not site["is_observed"]
                    }.items()
                }

                if not all(list(sites_in_theta.values())):
                    missing_sites = [name for name, in_theta in sites_in_theta.items() if not in_theta]
                    warnings.warn(
                        f"Sites: {missing_sites} were not specified in theta. "
                        "Log-likelihood will not be fully defined by theta."
                        "Results should be independent of the given seed"
                    )


            if return_type == "joint-log-likelihood":
                return joint_log_density
            
            
            prior_loglik = {
                name: site["fn"].log_prob(site["value"])
                for name, site in trace.items()
                if site["type"] == "sample" and not site["is_observed"]
            }

            data_loglik = {
                name: site["fn"].log_prob(site["value"])
                for name, site in trace.items()
                if site["type"] == "sample" and site["is_observed"]
            }

            if return_type == "full":
                return joint_log_density, prior_loglik, data_loglik

            prior_loglik_sum = {
                key: np.sum(value) for key, value in prior_loglik.items()
            }

            data_loglik_sum = {
                key: np.sum(value) for key, value in data_loglik.items()
            }

            if return_type == "summed-by-site":
                return joint_log_density, prior_loglik_sum, data_loglik_sum

            prior_loglik_total = np.sum(list(prior_loglik_sum.values()))
            data_loglik_total = np.sum(list(data_loglik_sum.values()))
            
            if return_type == "summed-by-prior-data":
                return joint_log_density, prior_loglik_total, data_loglik_total
            
            raise NotImplementedError(f"return_type flag: {return_type} is not implemented")


        return log_density

    @lru_cache
    def prior_predictions(self, n=100, seed=1):
        key = jax.random.PRNGKey(seed)
        obs, masks = self.observation_parser()

        model_kwargs = self.preprocessing(
            obs=obs, 
            masks=masks,
        )
        
        # prepare model
        model = partial(
            self.inference_model, 
            solver=self.evaluator, 
            **model_kwargs
        )    
   
        prior_predictive = Predictive(
            model, num_samples=n, batch_ndims=2
        )
        prior_predictions = prior_predictive(key)

        loglik = numpyro.infer.log_likelihood(
            model=model, 
            posterior_samples=prior_predictions, 
            batch_ndims=2, 
            obs=obs, 
            masks=masks
        )

        preds = self.simulation.data_variables
        preds_obs = [f"{d}_obs" for d in self.simulation.data_variables]
        priors = list(self.simulation.model_parameter_dict.keys())
        # TODO: This causes priors to be dropped that are not part of the 
        #       config file. Makes interactive model development more difficult
        #       This should be derived from the model directly.
        posterior_coords = self.posterior_coordinates
        posterior_coords["draw"] = list(range(n))
        data_structure = self.posterior_data_structure
        
        idata = az.from_dict(
            observed_data=obs,
            prior={k: v for k, v in prior_predictions.items() if k in priors},
            prior_predictive={k: v for k, v in prior_predictions.items() if k in preds+preds_obs},
            log_likelihood=loglik,
            dims=data_structure,
            coords=posterior_coords,
        )

        return idata
        # self.idata.to_netcdf(f"{self.simulation.output_path}/numpyro_prior_predictions.nc")
    

    def svi_posterior(self, svi_result, model, guide, key, n=1000):
        key, *subkeys = jax.random.split(key, 4)
        keys = iter(subkeys)

        obs, masks = self.observation_parser()

        params = svi_result.params

        # predictive = Predictive(
        #     model, guide=guide, params=params, 
        #     num_samples=n, batch_ndims=2
        # )
        # samples = predictive(next(keys))    

        predictive = Predictive(
            guide, params=params, 
            num_samples=n, batch_ndims=2
        )
        posterior_samples = predictive(next(keys))

        predictive = Predictive(
            model, posterior_samples, params=params, 
            num_samples=n, batch_ndims=2
        )
        posterior_predictions = predictive(next(keys))


        loglik = numpyro.infer.log_likelihood(
            model=model, 
            posterior_samples=posterior_samples, 
            batch_ndims=2, 
        )

        preds = self.simulation.data_variables
        preds_obs = [f"{d}_obs" for d in self.simulation.data_variables]
        priors = list(self.simulation.model_parameter_dict.keys())
        posterior_coords = self.posterior_coordinates
        posterior_coords["draw"] = list(range(n))
        data_structure = self.posterior_data_structure

        # TODO add prior data structure. Address this when a proper coordinate
        # dimensionality backend is implemented (config module)
        
        idata = az.from_dict(
            observed_data=obs,
            posterior={k: v for k, v in posterior_predictions.items() if k in priors},
            posterior_predictive={k: v for k, v in posterior_predictions.items() if k in preds},
            log_likelihood=loglik,
            dims=data_structure,
            coords=posterior_coords,
        )

        return idata
    

    @staticmethod
    def get_dict(group: xr.Dataset):
        data_dict = group.to_dict()["data_vars"]
        return {k: np.array(val["data"]) for k, val in data_dict.items()}


    @lru_cache
    def posterior_predictions(self, n: Optional[int]=None, seed=1):
        # TODO: It may be necessary that the coordinates should be passed as 
        # constant data. Because if the model is compiled with them once, 
        # according to the design philosophy of JAX, the model will not 
        # be evaluated again. But considering that the jitted functions do take
        # coordinates as an input argument, maybe I'm okay. This should be
        # tested.
        posterior = self.idata.posterior
        stacked_posterior = posterior.stack(sample=("chain", "draw"))
        n_samples = len(stacked_posterior.sample)
    
        if n is not None:
            key = jax.random.PRNGKey(seed)
            selection = jax.random.choice(key=key, a=jnp.array((range(n_samples))), replace=False, shape=(n, ))
            stacked_posterior = stacked_posterior.isel(sample=selection)


        preds = []
        for i in stacked_posterior.sample:
            sample = i.values.tolist()
            chain, draw = sample
            theta = stacked_posterior.sel(sample=sample)
            evaluator = self.simulation.dispatch(theta=self.get_dict(theta))
            evaluator()
            ds = evaluator.results

            ds = ds.assign_coords({"chain": chain, "draw": draw})
            ds = ds.expand_dims(("chain", "draw"))
            preds.append(ds)


        # key = jax.random.PRNGKey(seed)
        # model = partial(self.model, solver=self.evaluator)    
        # predict = numpyro.infer.Predictive(model, posterior_samples=posterior, batch_ndims=2)
        # predict(key, obs=obs, masks=masks)
        

        return xr.combine_by_coords(preds)

    def store_results(self, output=None):
        if output is not None:
            self.idata.to_netcdf(output)
        else:
            self.idata.to_netcdf(f"{self.simulation.output_path}/numpyro_posterior.nc")

    def load_results(self, file="numpyro_posterior.nc"):
        self.idata = az.from_netcdf(f"{self.simulation.output_path}/{file}")


    def plot(self):
        # TODO: combine prior_predictions and posterior predictions
        if hasattr(self.idata, "posterior"):
            axes = az.plot_trace(
                self.idata,
                var_names=self.simulation.model_parameter_names
            )
            fig = plt.gcf()
            fig.savefig(f"{self.simulation.output_path}/trace.png")
            plt.close()
            axes = az.plot_pair(
                self.idata, 
                divergences=True, 
                var_names=self.simulation.model_parameter_names
            )
            fig = plt.gcf()
            fig.savefig(f"{self.simulation.output_path}/pairs_posterior.png")
            plt.close()

        if hasattr(self.idata, "prior_predictive"):
            self.plot_prior_predictive()

        plot_func = getattr(self.simulation, self.plot_function)
        plot_func()

    def plot_prior_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={}, 
            n=None, seed=None
        ):
        if n is None:
            n = self.n_predictions

        if seed is None:
            seed = self.simulation.seed
        
        idata = self.prior_predictions(
            n=n, 
            # seed only controls the parameters samples drawn from posterior
            seed=seed
        )
        plot_func = getattr(self.simulation, self.plot_function)
        
        ax = self.plot_predictions(
            observations=self.simulation.observations,
            predictions=idata.prior_predictive,
            data_variable=data_variable,
            x_dim=x_dim,
            ax=ax,
            subset=subset,
            mode="draws"
        )

        return ax

    def plot_posterior_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={},
            n=None, seed=None
        ):
        # TODO: This method should be trashed. It is not really useful
        if n is None:
            n = self.n_predictions
        
        if seed is None:
            seed = self.simulation.seed
        
        predictions = self.posterior_predictions(
            n=n, 
            # seed only controls the parameters samples drawn from posterior
            seed=seed
        )
        
        ax = self.plot_predictions(
            observations=self.simulation.observations,
            predictions=predictions,
            data_variable=data_variable,
            x_dim=x_dim,
            ax=ax,
            subset=subset,
        )

        return ax



    def plot_predictions(
            self, 
            observations,
            predictions,
            data_variable: str, 
            x_dim: str, 
            ax=None, 
            subset={},
            mode="mean+hdi",
            plot_options={"obs": {}, "pred_mean": {}, "pred_draws": {}, "pred_hdi": {}},
        ):
        # filter subset coordinates present in data_variable
        subset = {k: v for k, v in subset.items() if k in observations.coords}
        
        # select subset
        obs = observations.sel(subset)[data_variable]
        preds = predictions.sel(subset)[f"{data_variable}"]
        
        # stack all dims that are not in the time dimension
        if len(obs.dims) == 1:
            # add a dummy batch dimension
            obs = obs.expand_dims("batch")
            obs = obs.assign_coords(batch=[0])

            preds = preds.expand_dims("batch")
            preds = preds.assign_coords(batch=[0])


        stack_dims = [d for d in obs.dims if d != x_dim]
        obs = obs.stack(i=stack_dims)
        preds = preds.stack(i=stack_dims)
        N = len(obs.coords["i"])
            
        hdi = az.hdi(preds, .95)[f"{data_variable}"]

        if ax is None:
            ax = plt.subplot(111)
        
        y_mean = preds.mean(dim=("chain", "draw"))

        for i in obs.i:
            if obs.sel(i=i).isnull().all():
                # skip plotting combinations, where all values are NaN
                continue
            
            if mode == "mean+hdi":
                kwargs_hdi = dict(color="black", alpha=0.1)
                kwargs_hdi.update(plot_options.get("pred_hdi", {}))
                ax.fill_between(
                    preds[x_dim].values, *hdi.sel(i=i).values.T, 
                    **kwargs_hdi
                )

                kwargs_mean = dict(color="black", lw=1, alpha=max(1/N, 0.05))
                kwargs_mean.update(plot_options.get("pred_mean", {}))
                ax.plot(
                    preds[x_dim].values, y_mean.sel(i=i).values, 
                    **kwargs_mean
                )
            elif mode == "draws":
                kwargs_draws = dict(color="black", lw=0.5, alpha=max(1/N, 0.05))
                kwargs_draws.update(plot_options.get("pred_draws", {}))
                ys = preds.sel(i=i).stack(sample=("chain", "draw"))
                ax.plot(
                    preds[x_dim].values, ys.values, 
                    **kwargs_draws
                )
            else:
                raise NotImplementedError(
                    f"Mode '{mode}' not implemented. "
                    "Choose 'mean+hdi' or 'draws'."
                )

            kwargs_obs = dict(marker="o", ls="", ms=3, color="tab:blue")
            kwargs_obs.update(plot_options.get("obs", {}))
            ax.plot(
                obs[x_dim].values, obs.sel(i=i).values, 
                **kwargs_obs
            )
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)

        return ax

    # This is a separate script!    
    def combine_chains(self, chain_location="chains", drop_extra_vars=[]):
        """Combine chains if chains were computed in a fully parallelized manner
        (on different machines, jobs, etc.). 

        In addition, the method drops all data variables and *_norm priors 
        (i.e. helper priors with a normal base). This is done, in order to
        create slim data objects for storage.

        Parameters
        ----------
        chain_location : str, optional
            location of the chains, relative to the simulation.output_path, this
            parameter is simulteneously the string appended to the saved 
            posterior. By default "chains"
        drop_extra_vars : List, optional
            any additional variables to drop from the posterior
        """
        sim = self.simulation
        pseudo_chains = glob.glob(
            f"{sim.output_path}/{chain_location}/*/numpyro_posterior.nc"
        )

        # just be aware that in the case of MAP this is not an acutal posterior.
        # But it can behave like one with multiple chains (1 for each start)
        idata = az.from_netcdf(pseudo_chains[0])
        posterior = idata.posterior
        log_likelihood = idata.log_likelihood

        # iterate over the posterior files with a progress bar (depending on the
        # size and number of posteriors this op needs time and memory)
        tqdm_iterator = tqdm(
            enumerate(pseudo_chains, start=1), 
            total=len(pseudo_chains),
            desc="Concatenating posteriors"
        )
        for i, f in tqdm_iterator:
            idata = az.from_netcdf(f)
            ccord = {"chain": np.array([i])}
            
            # add chain coordinate to posterior and likelihood
            idata.posterior = idata.posterior.assign_coords(ccord)
            idata.log_likelihood = idata.log_likelihood.assign_coords(ccord)

            # concatenate chains
            posterior = xr.concat([posterior, idata.posterior], dim="chain")
            log_likelihood = xr.concat(
                [log_likelihood, idata.log_likelihood], 
                dim="chain"
            )

        # drop extra variables if they are included in the posterior
        drop_vars = [k for k in list(posterior.data_vars.keys()) if "_norm" in k]
        drop_vars = drop_vars + sim.data_variables + drop_extra_vars
        drop_vars = [v for v in drop_vars if v in posterior]
        drop_coords = [c for c in list(posterior.coords.keys()) if c.split("_dim_")[0] in drop_vars]
        posterior = posterior.drop(drop_vars)
        posterior = posterior.drop(drop_coords)

        posterior = rename_extra_dims(
            posterior, 
            new_dim="substance", 
            new_coords=sim.observations.attrs["substance"]
        )

        # store mutlichain inferencedata to the main output directory
        # this is also a slim posterior that only contains the necessary information
        # posterior and likelihood and has therefore a small file size
        idata_multichain = az.InferenceData(
            posterior=posterior, 
            log_likelihood=log_likelihood
        )

        idata_multichain.to_netcdf(
            f"{sim.output_path}/numpyro_posterior_{chain_location}.nc"
        )

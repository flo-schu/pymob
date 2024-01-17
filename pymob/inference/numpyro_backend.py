from functools import partial, lru_cache
import re
from typing import Tuple, Dict, Union, Optional, Callable
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


sympy2jax_extra_funcs = {
    sympy.Array: jnp.array,
    sympy.Tuple: tuple,
}

def LogNormalTrans(loc, scale):
    return TransformedDistribution(
        Normal(0,1), 
        [
            transforms.AffineTransform(loc=jnp.log(loc), scale=scale), 
            exp()
        ]
    )


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
            "halfnorm": dist.HalfNormal,
            "poisson": (dist.Poisson, {"mu": "rate"}),
        }
        
        self.simulation = simulation
        self.evaluator = self.parse_deterministic_model()
        self.prior = self.parse_model_priors()
        self.error_model = self.parse_error_model()
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
    def n_predictions(self):
        return self.simulation.config.getint(
            "inference", "n_predictions", fallback=None
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

    def run(self):
        # set parameters of JAX and numpyro
        # jax.config.update("jax_enable_x64", True)
        numpyro.set_host_device_count(self.chains)

        # generate random keys
        key = jax.random.PRNGKey(1)
        key, *subkeys = jax.random.split(key, 20)
        keys = iter(subkeys)

        # parse observations and masks for missing data
        obs, masks = self.observation_parser()

        # prepare model and print information about shapes
        model = partial(
            self.inference_model, 
            solver=self.evaluator, 
            obs=obs, 
            masks=masks
        )    

        with numpyro.handlers.seed(rng_seed=1):
            trace = numpyro.handlers.trace(model).get_trace()
        print(numpyro.util.format_shapes(trace))
        
        # set up kernel, MCMC        
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

        mcmc = infer.MCMC(
            sampler=kernel,
            num_warmup=self.warmup,
            num_samples=self.draws,
            num_chains=self.chains,
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
    
    def prior_predictive_checks(self, seed=1):
        key = jax.random.PRNGKey(seed)
        model = partial(self.inference_model, solver=self.evaluator)    
            
        obs, masks = self.observation_parser()

        prior_predictive = Predictive(model, num_samples=100, batch_ndims=2)
        prior_predictions = prior_predictive(key, obs=obs, masks=masks)

        loglik = numpyro.infer.log_likelihood(
            model=model, 
            posterior_samples=prior_predictions, 
            batch_ndims=2, 
            obs=obs, 
            masks=masks
        )

        preds = self.simulation.data_variables
        priors = list(self.simulation.model_parameter_dict.keys())
        posterior_coords = self.posterior_coordinates
        data_structure = self.posterior_data_structure
        
        idata = az.from_dict(
            observed_data=obs,
            prior={k: v for k, v in prior_predictions.items() if k in priors},
            prior_predictive={k: v for k, v in prior_predictions.items() if k in preds},
            log_likelihood=loglik,
            dims=data_structure,
            coords=posterior_coords,
        )

        self.idata = idata
        self.idata.to_netcdf(f"{self.simulation.output_path}/numpyro_prior_predictions.nc")
    
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

    def store_results(self):
        self.idata.to_netcdf(f"{self.simulation.output_path}/numpyro_posterior.nc")

    def load_results(self):
        self.idata = az.from_netcdf(f"{self.simulation.output_path}/numpyro_posterior.nc")

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

    def plot_prior_predictive(self):
        self.idata

    def plot_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={}
        ):
        obs = self.simulation.observations.sel(subset)
        
        post_pred = self.posterior_predictions(
            n=self.n_predictions, 
            # seed only controls the parameters samples drawn from posterior
            seed=self.simulation.seed
        ).sel(subset)

        hdi = az.hdi(post_pred, .95)

        if ax is None:
            ax = plt.subplot(111)
        
        y_mean = post_pred[data_variable].mean(dim=("chain", "draw"))
        ax.plot(
            post_pred[x_dim].values, y_mean.values, 
            color="black", lw=.8
        )

        ax.fill_between(
            post_pred[x_dim].values, *hdi[data_variable].values.T, 
            alpha=.5, color="grey"
        )

        ax.plot(
            obs[x_dim].values, obs[data_variable].values, 
            marker="o", ls="", ms=3
        )
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)

        return ax

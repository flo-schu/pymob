from functools import partial, lru_cache
from pymob import SimulationBase
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

def LogNormalTrans(loc, scale):
    return TransformedDistribution(
        Normal(0,1), 
        [
            transforms.AffineTransform(loc=jnp.log(loc), scale=scale), 
            exp()
        ]
    )

exp = transforms.ExpTransform
sigmoid = transforms.SigmoidTransform
C = transforms.ComposeTransform

DISTRIBUTION_MAPPER = {
    "lognorm": dist.LogNormal,
    "binom": dist.Binomial,
    "normal": dist.Normal,
}

class NumpyroBackend:
    def __init__(
        self, 
        simulation: SimulationBase
    ):
        self.simulation = simulation
        self.evaluator = self.model_parser()
        self.prior = ...
        self.distance_function = ...

        self.abc = None
        self.history = None
        self.posterior = None
        self.idata: az.InferenceData

    @property
    def plot_function(self):
        return self.simulation.config.get(
            "inference", "plot_function", 
            fallback=None
        )

    @property
    def extra_vars(self):
        extra = self.simulation.config.getlist(
            "inference", "extra_vars", fallback=None
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

    def model_parser(self):
        def evaluator(theta, seed=None):
            evaluator = self.simulation.dispatch(theta=theta)
            evaluator(seed)
            return evaluator.Y
        
        return evaluator

    def model(self):
        pass


    def observation_parser(self):
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
        
        # masking
        # m = jnp.nonzero(mask)[0]
        return observations, masks
    
    def generate_artificial_data(self, theta, key, nan_frac=0.2):  
        # create artificial data from Evaluator      
        y_sim = self.evaluator(theta)
        key, *subkeys = jax.random.split(key, 5)
        y_sim["cext"] = dist.LogNormal(loc=jnp.log(y_sim["cext"]), scale=0.1).sample(subkeys[0])
        y_sim["cint"] = dist.LogNormal(loc=jnp.log(y_sim["cint"]), scale=0.1).sample(subkeys[1])
        y_sim["nrf2"] = dist.LogNormal(loc=jnp.log(y_sim["nrf2"]), scale=0.1).sample(subkeys[2])
        y_sim["lethality"] = dist.Binomial(total_count=9, probs=y_sim["lethality"]).sample(subkeys[3]).astype(float)

        # add missing data
        masks = {}
        key, *subkeys = jax.random.split(key, 5)
        for i, (k, val) in enumerate(y_sim.items()):
            nans = dist.Bernoulli(probs=nan_frac).expand(val.shape).sample(subkeys[i])
            val = val.at[jnp.nonzero(nans)].set(jnp.nan)
            m = jnp.where(nans==1, False, True)
            masks.update({k: m})
            y_sim.update({k: val})

        return y_sim, masks

    @staticmethod
    def param_to_prior(par):
        parname = par.name
        distribution, cluttered_arguments = par.prior.split("(", 1)
        param_strings = cluttered_arguments.split(")", 1)[0].split(",")
        params = {}
        for parstr in param_strings:
            key, val = parstr.split("=")
            params.update({key:float(val)})

        return parname, distribution, params

    def param(self, name):
        return [p for p in self.simulation.free_model_parameters if p.name == name][0]

    @classmethod
    def parse_prior(cls, par):
        name, dist, params = cls.param_to_prior(par)
        distribution = DISTRIBUTION_MAPPER[dist]
        
        return numpyro.sample(name, distribution, **params)

    def parse_model(self):
        theta = {}
        for par in self.simulation.free_model_parameters:
            p_n, p_d, p_p = self.param_to_prior(par=par)
            pri = numpyro.sample(p_n, dist.LogNormal(loc=p_p["scale"], scale=p_p["s"]))
            
            theta.update({p_n: pri})

    @staticmethod
    def model(solver, obs, masks):
        EPS = 1e-8

        k_i = numpyro.sample("k_i", LogNormalTrans(loc=5, scale=1))
        r_rt = numpyro.sample("r_rt", LogNormalTrans(loc=0.1, scale=1))
        r_rd = numpyro.sample("r_rd", LogNormalTrans(loc=0.5, scale=1))
        v_rt = numpyro.sample("v_rt", LogNormalTrans(loc=1.0, scale=1))
        z_ci = numpyro.sample("z_ci", LogNormalTrans(loc=500.0, scale=1))
        r_pt = numpyro.sample("r_pt", LogNormalTrans(loc=0.1, scale=1))
        r_pd = numpyro.sample("r_pd", LogNormalTrans(loc=0.01, scale=1))
        z = numpyro.sample("z", LogNormalTrans(loc=2.0, scale=1))
        kk = numpyro.sample("kk", LogNormalTrans(loc=0.005, scale=1))
        b_base = numpyro.sample("b_base", LogNormalTrans(loc=0.1, scale=1))
        sigma_cint = numpyro.sample("sigma_cint", dist.HalfNormal(scale=0.1))
        sigma_cext = numpyro.sample("sigma_cext", dist.HalfNormal(scale=0.1))
        sigma_nrf2 = numpyro.sample("sigma_nrf2", dist.HalfNormal(scale=0.1))
        theta = {
            "k_i": k_i,
            "r_rt": r_rt,
            "r_rd": r_rd,
            "v_rt": v_rt,
            "z_ci": z_ci,
            "r_pt": r_pt,
            "r_pd": r_pd,
            "volume_ratio": jnp.inf,
            "z": z,
            "kk": kk,
            "b_base": b_base,
        }
        sim = solver(theta=theta)
        cext = numpyro.deterministic("cext", sim["cext"])
        cint = numpyro.deterministic("cint", sim["cint"])
        nrf2 = numpyro.deterministic("nrf2", sim["nrf2"])
        leth = numpyro.deterministic("lethality", sim["lethality"])

        numpyro.sample("cext_obs", LogNormalTrans(loc=cext + EPS, scale=sigma_cext).mask(masks["cext"]), obs=obs["cext"] + EPS)
        numpyro.sample("cint_obs", LogNormalTrans(loc=cint + EPS, scale=sigma_cint).mask(masks["cint"]), obs=obs["cint"] + EPS)
        numpyro.sample("nrf2_obs", LogNormalTrans(loc=nrf2 + EPS, scale=sigma_nrf2).mask(masks["nrf2"]), obs=obs["nrf2"] + EPS)
        numpyro.sample("lethality_obs", dist.Binomial(probs=leth, total_count=obs["nzfe"]).mask(masks["lethality"]), obs=obs["lethality"])


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
        model = partial(self.model, solver=self.evaluator, obs=obs, masks=masks)    

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
            init_strategy=infer.init_to_median
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
        model = partial(self.model, solver=self.evaluator)    
            
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
        posterior_coords = self.posterior_coords
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
    def posterior_predictions(self, n: int=None, seed=1):
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

    def create_idata_from_unlabelled(self):
        posterior = self.idata.posterior.to_dict()["data_vars"]
        likelihood = self.idata.log_likelihood.to_dict()["data_vars"]
        sample_stats = self.idata.sample_stats.to_dict()["data_vars"]
        obs, masks = self.observation_parser()

        posterior = {k: np.array(val["data"]) for k, val in posterior.items()}
        likelihood = {k: np.array(val["data"]) for k, val in likelihood.items()}
        sample_stats = {k: np.array(val["data"]) for k, val in sample_stats.items()}
        
        data_structure = self.simulation.data_structure.copy()
        data_structure_loglik = {f"{dv}_obs": dims for dv, dims in data_structure.items()}
        data_structure.update(data_structure_loglik)
        
        posterior_coords = self.simulation.coordinates.copy()
        posterior_coords.update({"draw": list(range(2000)), "chain": [0]})
        self.idata = az.from_dict(
            observed_data=obs,
            posterior={k: v for k, v in posterior.items() if k in priors},
            posterior_predictive={k: v for k, v in posterior.items() if k in preds},
            log_likelihood=likelihood,
            sample_stats=sample_stats,
            dims=data_structure,
            coords=posterior_coords,
        )

    def plot_prior_predictive(self):
        self.idata

import os
from functools import lru_cache
import tempfile

import numpy as np
import pyabc
import xarray as xr
import arviz as az
from matplotlib import pyplot as plt
from pathos import multiprocessing as mp

from pymob import SimulationBase


class PyabcBackend:
    def __init__(
        self, 
        simulation: SimulationBase
    ):
        self.simulation = simulation
        self.evaluator = self.model_parser()
        self.prior = self.prior_parser(simulation.free_model_parameters)
        self.distance_function = self.distance_function_parser()
        self.observations = simulation.observations

        self.abc = None
        self.history = None
        self.posterior = None

    @property
    def sampler(self):
        return self.simulation.config.get("inference.pyabc", "sampler")
    
    @property
    def population_size(self):
        return self.simulation.config.getint("inference.pyabc", "population_size")
    
    @property
    def minimum_epsilon(self):
        return self.simulation.config.getfloat("inference.pyabc", "minimum_epsilon")
    
    @property
    def min_eps_diff(self):
        return self.simulation.config.getfloat("inference.pyabc", "min_eps_diff")
    
    @property
    def max_nr_populations(self):
        return self.simulation.config.getint("inference.pyabc", "max_nr_populations")
    
    @property
    def database(self):
        tmp = tempfile.gettempdir()
        dbp = self.simulation.config.get(
            "inference.pyabc", "database_path", fallback=f"{tmp}/pyabc.db"
        )
        if os.path.isabs(dbp):
            return dbp
        else:
            return os.path.join(self.simulation.output_path, dbp)
    
    @property
    def redis_password(self):
        return self.simulation.config.get("inference.pyabc.redis", "password", "nopassword")
    
    @property
    def redis_port(self):
        return self.simulation.config.getint("inference.pyabc.redis", "port", 1111)
    
    @property
    def history_id(self):
        return self.simulation.config.getint(
            "inference.pyabc", "eval.history_id", fallback=-1
        )

    @property
    def model_id(self):
        return self.simulation.config.getint(
            "inference.pyabc", "eval.model_id", fallback=0
        )
    
    @property
    def n_predictions(self):
        return self.simulation.config.getint(
            "inference.pyabc", "eval.n_predictions", fallback=50
        )
    
    @staticmethod
    def prior_parser(free_model_parameters: list):

        prior_dict = {}
        for mp in free_model_parameters:
            parname = mp.name
            distribution, cluttered_arguments = mp.prior.split("(", 1)
            param_strings = cluttered_arguments.split(")", 1)[0].split(",")
            params = {}
            for parstr in param_strings:
                key, val = parstr.split("=")
                params.update({key:float(val)})

            prior = pyabc.RV(distribution, **params)
            prior_dict.update({parname: prior})

        return pyabc.Distribution(**prior_dict)

    @property
    def plot_function(self):
        return self.simulation.config.get(
            "inference.pyabc", "plot_function", 
            fallback=None
        )

    def plot(self):
        plot_func = getattr(self.simulation, self.plot_function)
        plot_func()
        

    def model_parser(self):
        def model(theta):
            Y = self.simulation.evaluate(theta)
            return {"sim_results": Y}
        return model
    
    def distance_function_parser(self):
        def distance_function(x, x0):
            Y = x["sim_results"]
            obj_name, obj_value = self.simulation.objective_function(results=Y)
            return obj_value
        
        return distance_function

    def run(self):
        n_cores = self.simulation.n_cores
        print(f"Using {n_cores} CPU cores", flush=True)

        # before launch server in bash with `redis-server --port 1803`
        if self.sampler.lower() == "RedisEvalParallelSampler".lower():
            abc_sampler = pyabc.sampler.RedisEvalParallelSampler(
                host="localhost", 
                password=self.redis_password, 
                port=self.redis_port
            )

        elif self.sampler.lower() == "SingleCoreSampler".lower():
            abc_sampler = pyabc.sampler.SingleCoreSampler()

        elif self.sampler.lower() == "MulticoreParticleParallelSampler".lower():
            abc_sampler = pyabc.sampler.MulticoreParticleParallelSampler(
                n_procs=n_cores
            )

        elif self.sampler.lower() == "MulticoreEvalParallelSampler".lower():
            abc_sampler = pyabc.sampler.MulticoreEvalParallelSampler(
                n_procs=n_cores
            )

        else:
            raise NotImplementedError(
                "Sampler is not implemented. Choose one of: 'RedisEvalParallelSampler', " +
                "'SingleCoreSampler'"
            )


        self.abc = pyabc.ABCSMC(
            models=self.evaluator, 
            parameter_priors=self.prior, 
            distance_function=self.distance_function, 
            sampler=abc_sampler,
            population_size=self.population_size
        )

        self.history = self.abc.new("sqlite:///" + self.database)

        self.abc.run(
            minimum_epsilon=self.minimum_epsilon,
            min_eps_diff=self.min_eps_diff,
            max_nr_populations=self.max_nr_populations
        )


    def load_results(self):
        if self.history is None:
            self.history = pyabc.History(f"sqlite:///" + self.database)
        
        # set history id
        db_id = self.history_id
        self.history.id = self.history._find_latest_id() if db_id == -1 else db_id
        
        mod_id = self.model_id
        samples, w = self.history.get_distribution(m=mod_id, t=self.history.max_t)
        
        # re-sort parameters based on prior order
        samples = samples[self.prior.keys()]

        posterior = xr.DataArray(
            samples.values.reshape((1,*samples.values.shape)),
            coords=dict(
                chain=[1],
                draw=range(len(samples)),
                parameter=list(samples.columns)
            )
        )
        
        # posterior
        self.posterior = Posterior(posterior)


    def plot_chains(self):

        distributions = []
        for t in range(self.history.max_t + 1):
        # for t in range(2):
            print(f"iteration: {t}", end="\r")
            par_values, _ = self.history.get_distribution(m=0, t=t)
            post = par_values.to_xarray().to_array()
            post["id"] = range(len(post.id))
            post = post.assign_coords(iteration=t)
            distributions.append(post)

        trajectory = xr.concat(distributions, dim="iteration")
        
        parameters = trajectory.coords["variable"].values
        nit = len(trajectory.iteration)
        color="tab:blue"
        Np = len(parameters)
        fig, axes = plt.subplots(nrows=Np, ncols=2, figsize=(10, 2*Np),
            gridspec_kw=dict(width_ratios=[1, 1]))
        
        if len(axes.shape) == 1:
            axes = axes.reshape((Np, 2))
        
        for i, par in enumerate(parameters):
            ax = axes[i, 0]

            ax.plot(trajectory.iteration, trajectory.sel(variable=par), 
                color=color,
                # s=5,
                alpha=.01)
                
            # takes the last 
            ax2 = axes[i, 1]
            ax2.hist(trajectory.sel(variable=par, iteration=nit-1), 
                    alpha=.85, color=color)
            ax2.set_ylabel(f"{par.split('.')[-1].replace('_', ' ')}")
            
            ax.set_xlabel("iteration")
            ax.set_yscale("log")
            ax.set_ylabel(f"{par.split('.')[-1].replace('_', ' ')}")

        fig.subplots_adjust(wspace=.3, hspace=.4)

        return fig

    def store_results(self):
        """results are stored by default in database"""
        pass
   
    @lru_cache
    def make_posterior_predictions(self, n=50, seed=1):
        rng = np.random.default_rng(seed)
        post = self.posterior
        total_samples = post.samples.shape[1]

        # draw samples from posterior
        posterior_samples = rng.choice(post.samples.draw, size=n, replace=False)

        def predict(posterior_sample_id):
            params = post.draw(i=posterior_sample_id)
            res = self.evaluator(params.to_dict())
            res = self.simulation.results_to_df(res["sim_results"])
            res = res.assign_coords({"draw": posterior_sample_id, "chain": 1})
            res["params"] = params.samples
            res = res.expand_dims(("chain", "draw"))
            return res
        
        if self.simulation.n_cores == 1:
            results = list(map(predict, posterior_samples))
        else:
            with mp.ProcessingPool(self.simulation.n_cores) as pool:        
                results = pool.map(predict, posterior_samples)
            
        return xr.combine_by_coords(results)

    def plot_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={}
        ):
        obs = self.simulation.observations.sel(subset)
        
        post_pred = self.make_posterior_predictions(
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

class Posterior:
    def __init__(self, samples):
        self.samples = samples

    def __repr__(self):
        return str(self.samples)

    def to_dict(self):
        theta = self.samples
        return {par:float(theta.sel(parameter=par)) 
            for par in theta.parameter.values}

    def draw(self, i):
        return Posterior(self.samples.sel(draw=i, chain=1))

    def mean(self):
        return Posterior(self.samples.mean(dim=("chain", "draw")))



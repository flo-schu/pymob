from functools import partial
from pymob import SimulationBase
import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions import Normal, transforms, TransformedDistribution
from numpyro import infer
from diffrax import (
    diffeqsolve, 
    Dopri5, 
    ODETerm, 
    SaveAt, 
    PIDController, 
)

import arviz as az

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
        # self.observations, self.masks = self.observation_parser()

        self.abc = None
        self.history = None
        self.posterior = None

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
            .transpose("id", "time", "substance")
        data_vars = self.simulation.data_variables

        masks = []
        observations = []
        for d in data_vars:
            o = jnp.array(obs[d].values)
            m = jnp.logical_not(jnp.isnan(o))
            observations.append(o)
            masks.append(m)
        
        # masking
        # m = jnp.nonzero(mask)[0]
        return observations, masks
    
    def generate_artificial_data(self, theta, key, nan_frac=0.2):  
        # create artificial data from Evaluator      
        y_sim = self.evaluator(theta)
        key, *subkeys = jax.random.split(key, 5)
        y_sim[0] = dist.LogNormal(loc=jnp.log(y_sim[0]), scale=0.1).sample(subkeys[0])
        y_sim[1] = dist.LogNormal(loc=jnp.log(y_sim[1]), scale=0.1).sample(subkeys[1])
        y_sim[2] = dist.LogNormal(loc=jnp.log(y_sim[2]), scale=0.1).sample(subkeys[2])
        y_sim[3] = dist.Binomial(total_count=9, probs=y_sim[3]).sample(subkeys[3]).astype(float)

        # add missing data
        masks = []
        key, *subkeys = jax.random.split(key, 5)
        for i in range(4):
            nans = dist.Bernoulli(probs=nan_frac).expand(y_sim[i].shape).sample(subkeys[i])
            y_sim[i] = y_sim[i].at[jnp.nonzero(nans)].set(jnp.nan)
            m = jnp.where(nans==1, False, True)
            masks.append(m)

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
       
    def create_solver(self):
        from mod import tktd_rna_3, mappar

        @jax.jit
        def solver(theta, y0, t):
            """Solves the SIR model numerically

            A total population of 1000 persons is assumed. The initial value of susceptible
            persons is 1000 - init_i, whereas the initial number of infected persons is init_i.
            The solution is returned at the given meas_times.

            Args:
                init_i: Initial value of infected persons (state I)
                params: Array with parameters general death rate, cure/death rate,
                    general birth rate, infection rate (in this order)
                meas_times: Measurement time points

            Returns:
                Array with solution at the measurement time points (shape is n_meas x 3)
            """


            args = mappar(tktd_rna_3, theta, exclude=["X", "t"]) 
            f = lambda t, y0, params: tktd_rna_3(t, y0, *params)
        
            term = ODETerm(f)
            solver = Dopri5()
            saveat = SaveAt(ts=t)
            stepsize_controller = PIDController(rtol=1e-6, atol=1e-7)

            sol = diffeqsolve(
                term, solver, 
                t0=24, t1=120.0, dt0=0.1, 
                y0=y0, 
                saveat=saveat,
                stepsize_controller=stepsize_controller, 
                args=args,
                max_steps=10**6
            )
            return sol.ys
        return solver

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
        # volume_ratio = numpyro.sample("volume_ratio", dist.LogNormal(loc=jnp.log(5000), scale=1))
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
            # "z": 2.0,
            # "kk": 0.02,
            # "b_base": 0.1,
        }
        sim = solver(theta=theta)
        cext = numpyro.deterministic("cext", sim[0])
        cint = numpyro.deterministic("cint", sim[1])
        nrf2 = numpyro.deterministic("nrf2", sim[2])
        leth = numpyro.deterministic("lethality", sim[3])

        # cext = numpyro.sample("cext", dist.LogNormal(loc=jnp.log(y[0]), scale=sigma_cext).mask(mask["cext"].values), obs=obs["cext"].values)
        # cint = numpyro.sample("cint", dist.LogNormal(loc=jnp.log(y[1]), scale=sigma_cint).mask(mask["cint"].values), obs=obs["cint"].values)
        numpyro.sample("cext_obs", LogNormalTrans(loc=cext + EPS, scale=sigma_cext).mask(masks[0]), obs=obs[0] + EPS)
        numpyro.sample("cint_obs", LogNormalTrans(loc=cint + EPS, scale=sigma_cint).mask(masks[1]), obs=obs[1] + EPS)
        numpyro.sample("nrf2_obs", LogNormalTrans(loc=nrf2 + EPS, scale=sigma_nrf2).mask(masks[2]), obs=obs[2] + EPS)
        numpyro.sample("lethality_obs", dist.Binomial(probs=leth, total_count=9).mask(masks[3]), obs=obs[3])


    def run(self):
        # jax.config.update("jax_enable_x64", True)

        n_chains = 1
        numpyro.set_host_device_count(n_chains)

        # define parameters of the model
        theta = self.simulation.model_parameter_dict
        theta.update({"volume_ratio": 5000})
        coords = self.simulation.coordinates.copy()
        # self.simulation.coordinates["id"] = self.simulation.coordinates["id"][0:1]
        key = jax.random.PRNGKey(2)
        key, *subkeys = jax.random.split(key, 20)
        keys = iter(subkeys)

        # create artificial data from local solver
        # t = jnp.array(self.simulation.coordinates["time"])
        # y0 = (18.0, 0.0, 0.0, 0.0)
        # ode_sol = partial(solver, y0=y0, t=t)
        # y_sim = ode_sol(theta=theta)
        obs, masks = self.generate_artificial_data(theta, key=next(keys), nan_frac=0.0)
        # obs, masks = self.observation_parser()

        # real observations

        # bind the solver to the numpyro model
        # model = partial(model, solver=ode_sol)    
        model = partial(self.model, solver=self.evaluator)    
        
        kernel = infer.NUTS(
            model, 
            dense_mass=True, 
            # inverse_mass_matrix=inverse_mass_matrix,
            step_size=0.001,
            adapt_mass_matrix=True,
            adapt_step_size=True,
            max_tree_depth=8,
            target_accept_prob=0.8,
            init_strategy=infer.init_to_median
        )

        # TODO: Try with init args
        mcmc = infer.MCMC(
            sampler=kernel,
            num_warmup=2000,
            num_samples=2000,
            num_chains=n_chains,
            progress_bar=True,
        )

        mcmc.run(next(keys), obs=obs, masks=masks)
        mcmc.print_summary()

        idata = az.from_numpyro(mcmc)
        self.idata = idata

    def store_results(self):
        self.idata.to_netcdf(f"{self.simulation.output_path}/numpyro_posterior.nc")

    def load_data(self):
        self.idata = az.from_netcdf(f"{self.simulation.output_path}/numpyro_posterior.nc")

    def plot(self):
        az.plot_trace(self.idata)
        az.plot_pair(self.idata, divergences=True)


    def variational_inference(self):
        model = partial(self.model.__func__, solver=self.evaluator)    
        obs, masks = self.observation_parser()

        n_chains = 1
        numpyro.set_host_device_count(n_chains)
        # Use Variational inference to reparameterize a model 
        guide = AutoBNAFNormal(
            model, hidden_factors=[8, 8]
        )
        svi = SVI(model, guide, Adam(0.003), Trace_ELBO())
        
        print("Start training guide...")
        key, *subkeys = jax.random.split(key, 3)
        svi_result = svi.run(subkeys[0], num_steps=10000, obs=obs, masks=masks)
        # guide_samples = guide.sample_posterior(
        #     subkeys[1], svi_result.params, sample_shape=(1000,)
        # )["x"].copy()


        print("\nStart NeuTra HMC...")
        neutra = NeuTraReparam(guide, svi_result.params)
        neutra_model = neutra.reparam(model)
        reparameterized_nuts_kernel = NUTS(neutra_model)

        mcmc = infer.MCMC(
            sampler=reparameterized_nuts_kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=n_chains,
            progress_bar=True,
        )

        mcmc.run(subkey[2], obs=obs, masks=masks)
        mcmc.print_summary()
        # these arguments are from the adaptation procedure and can be saved
        # and reused for new runs without warmup.

    def sample_from_initialized_matrix(self):
        model = partial(self.model.__func__, solver=self.evaluator)    
        inverse_mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix
        step_size = mcmc.last_state.adapt_state.step_size
        

        sampling_kernel = infer.NUTS(
            model, 
            dense_mass=False, 
            inverse_mass_matrix=inverse_mass_matrix,
            step_size=step_size,
            adapt_mass_matrix=True,
            adapt_step_size=True,
            max_tree_depth=8,
            target_accept_prob=0.8,
            init_strategy=infer.init_to_median
        )

        mcmc_sampling = infer.MCMC(
            sampler=sampling_kernel,
            num_warmup=100,
            num_samples=1000,
            num_chains=n_chains,
            progress_bar=True,
        )

        key, subkey = jax.random.split(key)
        # obs, masks = generate_artificial_data(key)

        mcmc_sampling.run(subkey, obs=obs, masks=masks)

        mcmc_sampling.print_summary()

        # also get point to initialize

# # these arguments are from the adaptation procedure and can be saved
# # and reused for new runs without warmup.
# inverse_mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix
# step_size = mcmc.last_state.adapt_state.step_size

# # also get point to initialize
# proposal = mcmc.last_state

# apprently also last state can be passed to the run method
# TODO: It would possibly be a good idea to split warmup and sampling 
#       and save the mass matrix if warump was successful. 
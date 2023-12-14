from functools import partial
from pymob import SimulationBase
import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist
from numpyro import infer
from diffrax import (
    diffeqsolve, 
    Dopri5, 
    Kvaerno5,
    ODETerm, 
    SaveAt, 
    PIDController, 
    RecursiveCheckpointAdjoint
)
            

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
        self.observations = simulation.observations

        self.abc = None
        self.history = None
        self.posterior = None

    def model_parser(self):
        def evaluator(theta):
            evaluator = self.simulation.dispatch(theta=theta)
            evaluator()
            return evaluator.Y
        
        return evaluator

    def model(self):
        pass

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
       

    def run(self):
        from mod import tktd_rna_3, mappar
        # jax.config.update("jax_enable_x64", True)

        n_chains = 1
        numpyro.set_host_device_count(n_chains)
        EPS = 1e-8

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


        def model(solver, obs):
            k_i = numpyro.sample("k_i", dist.LogNormal(loc=jnp.log(5), scale=1))
            # k_i = numpyro.sample("k_i", dist.Normal(loc=5, scale=10))
            r_rt = numpyro.sample("r_rt", dist.LogNormal(loc=jnp.log(0.1), scale=1))
            r_rd = numpyro.sample("r_rd", dist.LogNormal(loc=jnp.log(0.5), scale=1))
            v_rt = numpyro.sample("v_rt", dist.LogNormal(loc=jnp.log(1.0), scale=1))
            z_ci = numpyro.sample("z_ci", dist.LogNormal(loc=jnp.log(500.0), scale=1))
            r_pt = numpyro.sample("r_pt", dist.LogNormal(loc=jnp.log(0.1), scale=1))
            r_pd = numpyro.sample("r_pd", dist.LogNormal(loc=jnp.log(0.01), scale=1))
            volume_ratio = numpyro.sample("volume_ratio", dist.LogNormal(loc=jnp.log(5000), scale=1))
            # z = numpyro.sample("z", dist.LogNormal(loc=jnp.log(2.0), scale=s))
            # kk = numpyro.sample("kk", dist.LogNormal(loc=jnp.log(0.02), scale=s))
            # b_base = numpyro.sample("b_base", dist.LogNormal(loc=jnp.log(0.1), scale=s))
            # sigma_cint = numpyro.sample("sigma_cint", dist.LogNormal(loc=jnp.log(0.1), scale=s))
            # sigma_cext = numpyro.sample("sigma_cext", dist.LogNormal(loc=jnp.log(0.1), scale=s))
            # sigma_nrf2 = numpyro.sample("sigma_nrf2", dist.HalfNormal(scale=0.1))

            theta = {
                "k_i": k_i,
                "r_rt": r_rt,
                "r_rd": r_rd,
                "v_rt": v_rt,
                "z_ci": z_ci,
                "r_pt": r_pt,
                "r_pd": r_pd,
                "volume_ratio": volume_ratio,
                # "z": z,
                # "kk": kk,
                # "b_base": b_base,
                
                # "k_i": 1e17, # breaks the ODE
                # "r_rt": 0.1,
                # "r_rd": 0.5,
                # "v_rt": 1.0,
                # "z_ci": 500,
                # "r_pt": 0.1,
                # "r_pd": 0.01,
                # "volume_ratio": 5000,
                "z": 2.0,
                "kk": 0.02,
                "b_base": 0.1,
            }
            # theta = (k_i, 0.1, 0.5, 1.0 ,500, 0.1, 0.01, 5000)
            sim = solver(theta=theta)
            cext = numpyro.deterministic("cext", sim[0])
            cint = numpyro.deterministic("cint", sim[1])
            nrf2 = numpyro.deterministic("nrf2", sim[2])

            # cext = numpyro.sample("cext", dist.LogNormal(loc=jnp.log(y[0]), scale=sigma_cext).mask(mask["cext"].values), obs=obs["cext"].values)
            # cint = numpyro.sample("cint", dist.LogNormal(loc=jnp.log(y[1]), scale=sigma_cint).mask(mask["cint"].values), obs=obs["cint"].values)
            numpyro.sample("lp_cext", dist.LogNormal(loc=jnp.log(cext + EPS), scale=0.1), obs=obs[0] + EPS)
            numpyro.sample("lp_cint", dist.LogNormal(loc=jnp.log(cint + EPS), scale=0.1), obs=obs[1] + EPS)
            numpyro.sample("lp_nrf2", dist.LogNormal(loc=jnp.log(nrf2 + EPS), scale=0.1), obs=obs[2] + EPS)
            # leth = numpyro.sample("lethality", dist.Binomial(probs=y[3], total_count=nzfe).mask(mask["lethality"].values), obs=obs["lethality"].values)


        theta = self.simulation.model_parameter_dict
        theta.update({"volume_ratio": 5000})
        y0 = (18.0, 0.0, 0.0, 0.0)
        t = jnp.array(self.simulation.coordinates["time"])

        ode_sol = partial(solver, y0=y0, t=t)
        y_sim = ode_sol(theta=theta)

        # obs = self.observations.transpose("id", "time", "substance")
        # mask = obs.isnull()
        # n = len(self.simulation.coordinates["time"])
        # nzfe = jnp.tile(self.observations.nzfe.values, n).reshape((n, 50)).T
        # mcmc.posterior
        model = partial(model, solver=ode_sol)    
        # model = partial(self.model, solver=self.evaluator)    
        kernel = infer.NUTS(
            model, 
            dense_mass=False, 
            # inverse_mass_matrix=inverse_mass_matrix,
            step_size=0.001,
            adapt_mass_matrix=True,
            adapt_step_size=True,
            target_accept_prob=0.8,
            init_strategy=infer.init_to_uniform
        )

        # TODO: Try with init args
        mcmc = infer.MCMC(
            sampler=kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=n_chains,
            progress_bar=True,
        )

        mcmc.run(jax.random.PRNGKey(1), obs=y_sim)
        mcmc.print_summary()
        
        import arviz as az
        idata = az.from_numpyro(mcmc)

        idata.to_netcdf(f"{self.simulation.output_path}/numpyro_posterior.nc")
        idata.posterior

        
        az.plot_pair(idata, divergences=True)

        # these arguments are from the adaptation procedure and can be saved
        # and reused for new runs without warmup.
        inverse_mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix
        step_size = mcmc.last_state.adapt_state.step_size
        
        # also get point to initialize
        proposal = mcmc.last_state

        # apprently also last state can be passed to the run method
        # TODO: It would possibly be a good idea to split warmup and sampling 
        #       and save the mass matrix if warump was successful. 
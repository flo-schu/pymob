from functools import partial
from typing import Literal
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

from pymob import SimulationBase, Config
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import DataVariable

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
EPS = 1e-8

# load the basic TKTD RNA Pulse case study and use as a parent class for the
# hierarchical model
config = Config()
config.case_study.name = "tktd_rna_pulse"
config.case_study.package = "case_studies"
config.import_casestudy_modules(reset_path=True)
from tktd_rna_pulse.sim import SingleSubstanceSim2

class NomixHierarchicalSimulation(SingleSubstanceSim2):
    def initialize(self, input):
        super().initialize(input)
        self.set_fixed_parameters(None)
        
        # configure JaxSolver
        self.solver = JaxSolver
        self.config.jaxsolver.batch_dimension = "id"

        # use_numpyro_backend can be written into the intiialize method
        # self.use_numpyro_backend()
        
    def use_numpyro_backend(self, error_model=None, only_prior=False):
        # configure the Numpyro backend
        self.config.inference_numpyro.user_defined_preprocessing = None
        self.config.inference_numpyro.user_defined_probability_model = None
        
        self.set_inferer("numpyro")
        self.inferer.inference_model = partial( 
            self.inferer.inference_model, 
            user_error_model=error_model,
            only_prior=only_prior
        )

        # set the fixed parameters
        self.model_parameters["parameters"] = self.config.model_parameters\
            .fixed_value_dict

    def setup_data_structure_from_observations(self):
        self.setup()

        # select observations
        obs = [0]
        sim.observations = sim.observations.isel(id=obs)
        sim.observations.attrs["substance"] = list(
            np.unique(sim.observations.substance)
        )
        sim.set_y0()
        sim.indices["substance"] = sim.indices["substance"].isel(id=obs)


    def setup_data_structure_manually(
        self, 
        scenario: Literal[
            "data_structure_01_single_observation",
            "data_structure_02_replicated_observation",
            "data_structure_03_gradient_observation",
            "data_structure_04_unreplicated_multi_substance",
            "data_structure_05_replicated_multi_experiment",
        ] = "data_structure_01_single_observation"
    ):
        self.config.case_study.scenario = scenario
        self.config.create_directory("results")
        self.config.create_directory("scenario")

        # mark existing data variables as unobserved
        for _, datavar in self.config.data_structure.all.items():
            datavar.observed = False
            datavar.min = np.nan
            datavar.max = np.nan

        # copy data structure from survival to lethality
        self.config.data_structure.lethality =\
            self.config.data_structure.survival # type:ignore

        if scenario == "data_structure_01_single_observation":
            self.define_observations_unreplicated()

        elif scenario == "data_structure_02_replicated_observation":
            self.define_observations_replicated()

        elif scenario == "data_structure_03_gradient_observation":
            self.define_observations_replicated_gradient()

        elif scenario == "data_structure_04_unreplicated_multi_substance":
            self.define_observations_unreplicated_multiple_substances()

        elif scenario == "data_structure_05_replicated_multi_experiment":
            self.define_observations_replicated_multi_experiment()

        # set up coordinates
        self.coordinates["time"] = np.arange(0, 120)
        # self.coordinates["substance"] = "diuron"

        # define starting values
        self.config.simulation.y0 = [
            "cext=cext_nom", 
            "cint=Array([0])", 
            "nrf2=Array([1])", 
            "P=Array([0])", 
        ]

        y0 = self.parse_input("y0", reference_data=self.observations, drop_dims=["time"])
        self.model_parameters["y0"] = y0

        # define parameters

        # set the fixed parameters
        self.model_parameters["parameters"] = self.config.model_parameters\
            .fixed_value_dict

        # set up the solver
        self.config.simulation.solver = "JaxSolver"
        self.config.jaxsolver.batch_dimension = "id"

        self.validate()
        self.config.save(force=True)

    def decorate_results(self, results):
        """Convenience function to add attributes and coordinates to simulation
        results needed for other post-processing tasks (e.g. plotting)
        """
        results.attrs["substance"] = np.unique(results.substance)
        results = results.assign_coords({
            "cext_nom": self.model_parameters["y0"]["cext"]
        })
        return results

    def plot(self, results: xr.Dataset):
        if "substance" not in results.coords:
            results = results.assign_coords({"substance": self.observations.substance})
        if "cext_nom" not in results.coords:
            results = results.assign_coords({"cext_nom": self.observations.cext_nom})
        fig = self._plot.plot_simulation_results(results)


    def define_observations_unreplicated(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ], dims=("id"), coords={"id": [0]}),
            "cext_nom":  xr.DataArray([1000    ], dims=("id"), coords={"id": [0]}),
            "substance": xr.DataArray(["diuron"], dims=("id"), coords={"id": [0]})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0],
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * 5, dims=("id"), coords={"id": np.arange(5)}),
            "cext_nom":  xr.DataArray([1000    ] * 5, dims=("id"), coords={"id": np.arange(5)}),
            "substance": xr.DataArray(["diuron"] * 5, dims=("id"), coords={"id": np.arange(5)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * 5,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated_multi_experiment(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        n = 10
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * n, dims=("id"), coords={"id": np.arange(n)}),
            "cext_nom":  xr.DataArray([1000    ] * n, dims=("id"), coords={"id": np.arange(n)}),
            "substance": xr.DataArray(["diuron"] * n, dims=("id"), coords={"id": np.arange(n)}),
            "experiment": xr.DataArray(np.repeat([0,1], int(n/2)), dims=("id"), coords={"id": np.arange(n)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * n,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated_gradient(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        n = 5
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * n, dims=("id"), coords={"id": np.arange(n)}),
            "cext_nom":  xr.DataArray(np.logspace(2,4, n), dims=("id"), coords={"id": np.arange(n)}),
            "substance": xr.DataArray(["diuron"] * n, dims=("id"), coords={"id": np.arange(n)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * n,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_unreplicated_multiple_substances(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      , 10,10], dims=("id"), coords={"id": np.arange(3)}),
            "cext_nom":  xr.DataArray([1000    , 100,10], dims=("id"), coords={"id": np.arange(3)}),
            "substance": xr.DataArray(["diuron", "naproxen","diclofenac"], dims=("id"), coords={"id": np.arange(3)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0,1,2],
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }


def conditional_survival_error_model(theta, simulation_results, observations, masks):
    # indexing
    substance_idx = observations["substance_index"]
    sigma_cint_indexed = theta["sigma_cint"][substance_idx]
    sigma_nrf2_indexed = theta["sigma_nrf2"][substance_idx]

    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), observations["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), observations["nrf2"].shape)

    # error model
    S = jnp.clip(simulation_results["survival"], EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = observations["survivors_before_t"]
    S_mask = masks["survival"]
    obs_survival = observations["survival"]
    
    # calculate likelihoods
    numpyro.sample("cint_obs", dist.LogNormal(
            loc=jnp.log(simulation_results["cint"] + EPS),  # type: ignore
            scale=sigma_cint_ix_bc  # type: ignore
        ).mask(masks["cint"]),
        obs=observations["cint"]
    )
    
    numpyro.sample("nrf2_obs", dist.LogNormal(
            loc=jnp.log(simulation_results["nrf2"]),  # type: ignore
            scale=sigma_nrf2_ix_bc  # type: ignore
        ).mask(masks["nrf2"]), 
        obs=observations["nrf2"]
    )    
    
    numpyro.sample(
        "survival_obs", dist.Binomial(
            probs=S_cond_, 
            total_count=n_surv
        ).mask(S_mask), 
        obs=obs_survival
    )


def independent_survival_error_model(theta, simulation_results, observations, masks):
    # indexing
    substance_idx = observations["substance_index"]
    sigma_cint_indexed = theta["sigma_cint"][substance_idx]
    sigma_nrf2_indexed = theta["sigma_nrf2"][substance_idx]

    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), masks["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), masks["nrf2"].shape)

    
    # calculate likelihoods
    numpyro.sample("cint_obs", dist.LogNormal(
            loc=jnp.log(simulation_results["cint"] + EPS),  # type: ignore
            scale=sigma_cint_ix_bc  # type: ignore
        ).mask(masks["cint"]),
        obs=observations["cint"]
    )
    
    numpyro.sample("nrf2_obs", dist.LogNormal(
            loc=jnp.log(simulation_results["nrf2"]),  # type: ignore
            scale=sigma_nrf2_ix_bc  # type: ignore
        ).mask(masks["nrf2"]), 
        obs=observations["nrf2"]
    )    
    
    numpyro.sample(
        "survival_obs", dist.Binomial(
            probs=simulation_results["survival"], 
            total_count=observations["nzfe"]
        ).mask(masks["survival"]), 
        obs=observations["survival"]
    )


if __name__ == "__main__":
    cfg = "case_studies/hierarchical_ode_model/scenarios/testing/settings.cfg"
    # cfg = "case_studies/tktd_rna_pulse/scenarios/rna_pulse_3_6c_substance_specific/settings.cfg"
    sim = NomixHierarchicalSimulation(cfg)
    
    # TODO: this will become a problem once I try to load different extra
    # modules. The way to deal with this is to load modules as a list and try
    # to get them in hierarchical order
    sim.config.import_casestudy_modules()

    # sim.setup_data_structure_from_observations()
    sim.setup_data_structure_manually(
        scenario="data_structure_05_replicated_multi_experiment"
    )

    # run a simulation
    sim.dispatch_constructor()
    e = sim.dispatch(theta=sim.model_parameter_dict)
    e()
    sim.plot(e.results)

    # generate artificial data
    sim.dispatch_constructor()
    res = sim.generate_artificial_data(nan_frac=0.0)
    res_ = sim.lethality_to_conditional_survival(res)

    # mark data as observed
    sim.observations = res_
    # mark existing data variables as unobserved
    for key, datavar in sim.config.data_structure.all.items():
        if key in ["cint", "nrf2", "survival"]:
            datavar.observed = True

    sim.plot(res)

    # perform inference
    sim.config.jaxsolver.throw_exception = False
    sim.dispatch_constructor()
    sim.use_numpyro_backend(
        error_model=independent_survival_error_model,
        only_prior=False
    )
    sim.config.inference_numpyro.kernel = "nuts"
    sim.config.inference_numpyro.draws = 2000
    sim.config.inference_numpyro.svi_iterations = 1000

    pp = sim.inferer.prior_predictions()
    plot_kwargs = dict(mode="mean+hdi")
    sim.inferer.plot_prior_predictions("cint", x_dim="time", **plot_kwargs) # type:ignore
    sim.inferer.run()

    # next steps:
    # 1. add substance index to obsertvation like in dict -> dataset
    # 2. convert lethality to conditional probability survival notation
    # 3. Try numpyro plate notation and when batch dimension is returned
    #    Or extend dims if no batch is present



    # define a hierarchical error structure
    # check out murefi for this

    # the long form should always be used for the actual model calculations
    # unless wide form is actually required (i.e. vectors or matrices need)
    # to enter the ODE
    
    # currently I use the substance as an index for broadcasting the parameters
    # from a substance index to the long form.
    # multilevel index or something along these lines would be needed to 
    # bring a multilevel index into the long form.


    # currently parameters are at least broadcasted in the JaxSolver, but this
    # is not happening with the other solvers. 
    # Approach:
    # + Define a module that can handle parameter broadcasting automatically 
    #   during dispatch. This can be adapted from the JaxSolver.
    # + Solvers themselves should only handle the casting of the data to types
    #   they require.
    # + This would mean that it is ensured that parameter, y_0 and x_in shapes
    #   can be handled by the solver, because they have been broadcasted, and
    #   can be vectorized or iterated over.
    #
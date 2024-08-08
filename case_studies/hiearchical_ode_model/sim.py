from functools import partial
import numpy as np

from pymob import SimulationBase, Config
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import DataVariable

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
        
    def use_numpyro_backend(self):
        # configure the Numpyro backend
        self.set_inferer("numpyro")
        self.config.inference_numpyro.user_defined_preprocessing = None
        self.inferer.preprocessing = partial(         # type: ignore
            sim.inferer.preprocessing,                # type: ignore
            ci_max=self.config.model_parameters.fixed_value_dict["ci_max"] 
        ) 

        # set the fixed parameters
        self.model_parameters["parameters"] = self.config.model_parameters\
            .fixed_value_dict



if __name__ == "__main__":
    cfg = "case_studies/hierarchical_ode_model/scenarios/testing/settings.cfg"
    cfg = "case_studies/tktd_rna_pulse/scenarios/rna_pulse_3_6c_substance_specific/settings.cfg"
    sim = NomixHierarchicalSimulation(cfg)
    
    # TODO: this will become a problem once I try to load different extra
    # modules. The way to deal with this is to load modules as a list and try
    # to get them in hierarchical order
    sim.config.import_casestudy_modules()
    sim.setup()

    # select observations
    obs = [0]
    sim.observations = sim.observations.isel(id=obs)
    sim.observations.attrs["substance"] = list(np.unique(sim.observations.substance))
    sim.set_y0()
    sim.indices["substance"] = sim.indices["substance"].isel(id=obs)

    # run a simulation
    sim.dispatch_constructor()
    e = sim.dispatch(theta=sim.model_parameter_dict)
    e()
    e.results
    sim.plot(e.results)

    # generate artificial data
    sim.dispatch_constructor()
    res = sim.generate_artificial_data()
    sim.plot(res)

    # perform inference
    sim.use_numpyro_backend()
    sim.config.inference_numpyro.kernel = "map"
    sim.config.inference_numpyro.draws = 1000
    sim.inferer.run()



    from tktd_rna_pulse import mod as trpmod

    trpmod.tktd_rna_3_6c

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
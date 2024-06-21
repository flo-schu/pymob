import numpy as np
import xarray as xr
from mod import lotka_volterra, solve, solve_jax
from plot import plot_trajectory
from pymob.simulation import SimulationBase
import prob

class Simulation(SimulationBase):
    solver = solve_jax
    model = lotka_volterra
    prob = prob

    def initialize(self, input):
        self.model_parameters["parameters"] =  dict(
            alpha = 0.5,  # Prey birth rate
            beta = 0.02,  # Rate at which predators decrease prey population
            gamma = 0.3,  # Predator reproduction rate
            delta = 0.01,  # Predator death rate
        )
        if self.config.getint("simulation", "replicated"):
            rng = np.random.default_rng(1)
            self.model_parameters["y0"] = rng.integers(1, 50, size=(10, 2))
        else:
            self.model_parameters["y0"] = np.array([40, 9])  # initial population of prey and predator

        
        self.observations = xr.load_dataset(input[1])
        self.observations["wolves"] = self.observations["wolves"] + 1e-8
        self.observations["rabbits"] = self.observations["rabbits"] + 1e-8

        
    @staticmethod
    def parameterize(free_parameters: dict, model_parameters):
        """Should avoid using input arg but instead take a single dictionary as 
        an input. This also then provides an harmonized IO between model and 
        parameters, which in addition is serializable to json.

        model parameters is provided by `functools.partial` on model initialization
        """
        # Initial conditions and parameters
        y0 = model_parameters["y0"]
        parameters = model_parameters["parameters"]
        # mapping of parameters *theta* to the model parameters accessed by
        # the solver. This task is necessary for any model 
        parameters.update(free_parameters)

        return dict(y0=y0, parameters=parameters)

    def set_coordinates(self, input):
        if hasattr(self, "observations"):
            return self.observations.time.values
        
        # Time settings
        t_start = 0
        t_end = 200
        t_step = 0.1
        time = np.arange(t_start, t_end, t_step)

        return time
    
    def plot(self, results):
        fig = plot_trajectory(results)
        fig.savefig(f"{self.output_path}/trajectory.png")

# CURRENTLY UNUSABLE SEE https://github.com/flo-schu/pymob/issues/6
class ReplicatedSimulation(Simulation):
    def set_coordinates(self, input):
        if self.observations is not None:
            return self.observations.time.values
        
        # Time settings
        t_start = 0
        t_end = 200
        t_step = 0.1
        time = np.arange(t_start, t_end, t_step)

        id = np.arange(len(self.model_parameters["y0"]))


        return time, id

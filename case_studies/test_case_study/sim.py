import numpy as np
import xarray as xr
from mod import lotka_volterra, solve
from plot import plot_trajectory
from pymob import SimulationBase

class Simulation(SimulationBase):
    solver = solve
    model = lotka_volterra

    def initialize(self, input):
        self.model_parameters["parameters"] =  dict(
            alpha = 0.1,  # Prey birth rate
            beta = 0.02,  # Rate at which predators decrease prey population
            gamma = 0.3,  # Predator reproduction rate
            delta = 0.01,  # Predator death rate
        )
        self.model_parameters["y0"] = [40, 9]  # initial population of prey and predator
        self.observations = xr.load_dataset(input[1])
        
    @staticmethod
    def parameterize(free_parameters, model_parameters):
        """Should avoid using input arg but instead take a single dictionary as 
        an input. This also then provides an harmonized IO between model and 
        parameters, which in addition is serializable to json.
        """
        # Initial conditions and parameters
        y0 = model_parameters["y0"]
        parameters = model_parameters["parameters"]
        # mapping of parameters *theta* to the model parameters accessed by
        # the solver. This task is necessary for any model 
        parameters.update(free_parameters["parameters"])

        return dict(y0=y0, parameters=parameters)

    def set_coordinates(self, input):
        # Time settings
        t_start = 0
        t_end = 200
        t_step = 0.1
        time = np.arange(t_start, t_end, t_step)

        return time
    
    def plot(self):
        fig = plot_trajectory(self.results)
        fig.savefig(f"{self.output_path}/trajectory.png")

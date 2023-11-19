import numpy as np
import xarray as xr
from scipy.integrate import odeint
from mod import lotka_volterra
from plot import plot_trajectory
from pymob import SimulationBase

class Simulation(SimulationBase):
    def initialize(self, input):
        self.observations = xr.load_dataset(input[1])
        
    
    def parameterize(self, input, theta):
        # Initial conditions and parameters
        
        y0 = [40, 9]  # initial population of prey and predator

        model_parameters = dict(
            alpha = 0.1,  # Prey birth rate
            beta = 0.02,  # Rate at which predators decrease prey population
            gamma = 0.3,  # Predator reproduction rate
            delta = 0.01,  # Predator death rate
        )

        # mapping of parameters *theta* to the model parameters accessed by
        # the solver. This task is necessary for any model 
        model_parameters.update(theta)

        return y0, model_parameters

    def set_coordinates(self, input):
        # Time settings
        t_start = 0
        t_end = 200
        t_step = 0.1
        time = np.arange(t_start, t_end, t_step)

        return time
    
    def run(self):
        # accesses the current model parameters in standard form. An ordered
        # list of the Parameter class
        y0, params = self.model_parameters

        # if necessary any mapping from this class to the solver takes place
        
        t = self.coordinates["time"]
        # Solve the differential equations using odeint
        solution = odeint(lotka_volterra, y0, t, args=tuple(params.values()))
        Y = solution  # Transpose the solution array
        
        return Y
       
    def plot(self):
        fig = plot_trajectory(self.results)
        fig.savefig(f"{self.output_path}/trajectory.png")

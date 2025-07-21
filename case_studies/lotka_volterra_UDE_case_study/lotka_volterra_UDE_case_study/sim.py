from case_studies.lotka_volterra_case_study.lotka_volterra_case_study.sim import Simulation_v2
from case_studies.lotka_volterra_UDE_case_study.lotka_volterra_UDE_case_study.mod import Func
from pymob.solvers.diffrax import JaxSolver
import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax.random as jr

class UDESimulation(Simulation_v2):
    alpha = 1.3
    beta = 0.9
    gamma = 0.8
    delta = 1.8

    key = jr.PRNGKey(5678)
    data_key, model_key, loader_key = jr.split(key, 3)
    model = Func(10,10,key=model_key,theta_true=(alpha,beta,gamma))
    
    solver = JaxSolver

    def initialize(self, input):

        alpha = 1.3
        beta = 0.9
        gamma = 0.8
        delta = 1.8

        key = jr.PRNGKey(5678)
        data_key, model_key, loader_key = jr.split(key, 3)
        self.model = Func(10,10,key=model_key,theta_true=(alpha,beta,gamma))

        self.observations = xr.load_dataset(input[0])

        y0 = self.parse_input("y0", drop_dims=["time"])
        self.model_parameters["y0"] = y0

        self.model_parameters["parameters"] = self.config.model_parameters.value_dict


# class UDESimulation2(Simulation_v2):
#     solver = JaxSolver

#     def initialize(self, input):
#         self.observations = xr.load_dataset(input[0])

#         y0 = self.parse_input("y0", drop_dims=["time"])
#         self.model_parameters["y0"] = y0

#         self.model_parameters["parameters"] = self.config.model_parameters.value_dict

#         weights = returnWeightsList(self, 70)
#         bias = returnBiasList(self, 17)

#         key = jr.PRNGKey(5678)
#         data_key, model_key, loader_key = jr.split(key, 3)

#         func = Func(2,5,3,key=model_key,theta_true=jnp.array([self.model_parameters["parameters"]["alpha"],self.model_parameters["parameters"]["gamma"]]))
#         func = setFuncWeightsAndBias(func, weights, bias, key=model_key)

#         self.model = func


def returnWeightsList(self, noWeights):
    list = []
    for i in np.arange(noWeights):
        list.append(self.model_parameters["parameters"]["weight"+str(i)])
    return list

def returnBiasList(self, noBias):
    list = []
    for i in np.arange(noBias):
        list.append(self.model_parameters["parameters"]["bias"+str(i)])
    return list
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax
from typing import Callable
from pymob.utils.UDE import UDEBase, transformBiasBackwards, transformWeightsBackwards
    
class Func(UDEBase):

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 2
    mlp_out_size: int = 2
    mlp_activation: Callable = staticmethod(jnn.softplus)
    mlp_final_activation: Callable = staticmethod(lambda x: x)

    alpha: jax.Array
    delta: jax.Array
    
    @staticmethod
    def model(t, y, mlp, alpha, delta, ):
        prey, predator = y

        # input = x_in.evaluate(t)
        
        dprey_dt_ode = alpha * prey 
        dpredator_dt_ode = - delta * predator
        dprey_dt_nn, dpredator_dt_nn = mlp(y) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])

        dprey_dt = dprey_dt_ode + dprey_dt_nn
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return dprey_dt, dpredator_dt
    
    @staticmethod
    def loss(y_obs, y_pred):
        return (y_obs - y_pred)**2 + 1e-2*(y_pred**-1)
    
class Func1D(UDEBase):

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 1
    mlp_out_size: int = 1

    r: jax.Array

    @staticmethod
    def model(y, mlp, r):
        X = y
        
        dX_dt = r * X + mlp(y)

        return dX_dt
    
    @staticmethod
    def loss(y_obs, y_pred):
        return (y_obs - y_pred)**2
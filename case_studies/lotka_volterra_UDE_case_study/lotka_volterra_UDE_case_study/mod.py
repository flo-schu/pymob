import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax
from pymob.utils.UDE import UDEBase
    
class Func(UDEBase):

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 2
    mlp_out_size: int = 2

    alpha: jax.Array
    delta: jax.Array

    def __init__(self, params, weights=None, bias=None, *, key, **kwargs):
        self.init_MLP(weights, bias, key=key)
        self.init_params(params)
        
    def __call__(self, t, y):
        """
        Returns the growth rates of predator and prey depending on their current state.

        Parameters
        ----------
        t : scalar
            Just here to fulfill the requirements by diffeqsolve(). Has no effect and
            can be set to None.
        y : jax.ArrayImpl
            Array containing two values: the current abundance of prey and predator,
            respectively.

        Returns:
        --------
        jax.ArrayImpl
            An array containing the growth rates of prey and predators, respectively.
        """

        params = self.preprocess_params()

        prey, predator = y
        
        dprey_dt_ode = params["alpha"] * prey 
        dpredator_dt_ode = - params["delta"] * predator
        dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])

        dprey_dt = dprey_dt_ode + dprey_dt_nn
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return jnp.array([dprey_dt.astype(float),dpredator_dt.astype(float)])
    
class Func1D(UDEBase):

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 1
    mlp_out_size: int = 1

    r: jax.Array

    def __init__(self, params, weights=None, bias=None, *, key, **kwargs):
        self.init_MLP(weights, bias, key=key)
        self.init_params(params)
        
    def __call__(self, t, y):
        """
        Returns the growth rates of predator and prey depending on their current state.

        Parameters
        ----------
        t : scalar
            Just here to fulfill the requirements by diffeqsolve(). Has no effect and
            can be set to None.
        y : jax.ArrayImpl
            Array containing two values: the current abundance of prey and predator,
            respectively.

        Returns:
        --------
        jax.ArrayImpl
            An array containing the growth rates of prey and predators, respectively.
        """

        params = self.preprocess_params()

        X = y
        
        dX_dt = params["r"] * X + self.mlp(y)

        return jnp.array(dX_dt.astype(float))
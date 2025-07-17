from functools import partial
import equinox as eqx
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import jax

class Func(eqx.Module):
    theta_true: jnp.array
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key, theta_true, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )
        self.theta_true = theta_true

    def __call__(self, t, y, *args):

        prey, predator = y
        alpha, beta, gamma = jax.lax.stop_gradient(self.theta_true)
        dprey_dt_ode = alpha * prey - beta * prey * predator
        dpredator_dt_ode = gamma * prey * predator
        dpredator_dt_nn = self.mlp(y)

        dprey_dt = dprey_dt_ode
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return jnp.array([dprey_dt.astype(float), dpredator_dt.astype(float)[0]])
    
    def __hash__(self):
        return 0
    
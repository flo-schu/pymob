import numpyro
import jax.numpy as jnp


def dummy_preprocessing(obs, masks):
    return {"obs": obs, "masks": masks}

def parameter_only_model(solver, obs, masks):
    numpyro.sample("alpha", fn=numpyro.distributions.LogNormal(jnp.log(0.5), 1))
    numpyro.sample("beta", fn=numpyro.distributions.LogNormal(jnp.log(0.02), 1))

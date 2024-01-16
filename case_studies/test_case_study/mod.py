from functools import partial
from pymob.sim.solvetools import mappar
from scipy.integrate import solve_ivp
import jax.numpy as jnp
import jax
from diffrax import (
    diffeqsolve, 
    Dopri5, 
    Kvaerno5,
    ODETerm, 
    SaveAt, 
    PIDController, 
    RecursiveCheckpointAdjoint
)

def solve(model, parameters, coordinates, data_variables):
    """Initial value problems always need the same number of recurrent arguments

    - parameters: define the model
    - y0: set the initial values of the ODE states
    - coordinates: are needed to know over which values to integrate
    - seed: In case stochastic processes take place inside the model this is necessary
    
    In order to make things explicit, all information which is needed by the
    model needs to be specified in the function signature. 
    This also makes the solvers functionally oriented, a feature that helps the
    usability of models accross inference frameworks. Where functions should not
    have side effects.

    Additionally, passing arguments via the signature makes it easier to write
    up models in a casual way and only later embed them into more regulated
    structures such as pymob

    """
    odeargs = mappar(model, parameters["parameters"], exclude=["t", "y"])
    t = coordinates["time"]
    results = solve_ivp(
        fun=model,
        y0=parameters["y0"],
        t_span=(t[0], t[-1]),
        t_eval=t,
        args=odeargs,
    )
    return {data_var:y for data_var, y in zip(data_variables, results.y)}

def solve_jax(model, parameters, coordinates, data_variables):
    time = jnp.array(coordinates["time"])
    params = parameters["parameters"]
    y0 = parameters["y0"]
    ode_args = mappar(model, params, exclude=["t", "y"])

    result = odesolve(model, tuple(y0), time, ode_args)
    res_dict = {v:val for v, val in zip(data_variables, result)}
    return res_dict

@partial(jax.jit, static_argnames=["model"])
def odesolve(model, y0, time, args):
    f = lambda t, y, args: model(t, y, *args)
    
    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=time)
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-7)

    sol = diffeqsolve(
        terms=term, 
        solver=solver, 
        t0=time.min(), 
        t1=time.max(), 
        dt0=0.1, 
        y0=tuple(y0), 
        args=args, 
        saveat=saveat, 
        stepsize_controller=stepsize_controller,
        adjoint=RecursiveCheckpointAdjoint(),
        max_steps=10**5,
        # throw=False returns inf for all t > t_b, where t_b is the time 
        # at which the solver broke due to reaching max_steps. This behavior
        # happens instead of throwing an exception.
        throw=False
    )
    
    return list(sol.ys)


def lotka_volterra(t, y, alpha, beta, gamma, delta):
    """
    Calculate the rate of change of prey and predator populations.

    Parameters:
    ----------
    y : array-like
        A list containing the current prey and predator populations [prey, predator].
    t : array-like
        Time points at which to evaluate the populations.
    alpha : float
        Prey birth rate.
    beta : float
        Rate at which predators decrease prey population.
    gamma : float
        Predator reproduction rate.
    delta : float
        Predator death rate.

    Returns:
    -------
    dydt : list
        Rate of change of prey and predator populations.
    """
    prey, predator = y
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return dprey_dt, dpredator_dt


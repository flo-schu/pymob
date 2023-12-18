from pymob.sim.solvetools import mappar
from scipy.integrate import odeint

def solve(model, parameters, coordinates, foo):
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

    return odeint(
        func=model,
        y0=parameters["y0"],
        t=coordinates["time"],
        args=odeargs,
    )


def lotka_volterra(y, t, alpha, beta, gamma, delta):
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
    return [dprey_dt, dpredator_dt]


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


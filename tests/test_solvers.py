import os
import time
import numpy as np

from pymob.solvers.diffrax import JaxSolver
from pymob.solvers.base import rect_interpolation, radius_interpolation, smoothed_interpolation, jump_interpolation
from tests.fixtures import init_simulation_casestudy_api, init_bufferguts_casestudy
from diffrax import Heun, Euler, Tsit5, Dopri5

from pymob.sim.evaluator import Evaluator
from pymob import SimulationBase

def test_benchmark_time():
    sim = init_simulation_casestudy_api()

    cpu_time_start = time.process_time()
    sim.benchmark(n=100)
    cpu_time_stop = time.process_time()

    t = cpu_time_stop - cpu_time_start

    if t > 1:
        raise AssertionError(f"Benchmarking took too long: {t}s. Expected t < 1s")


def test_benchmark_jaxsolver():
    sim = init_simulation_casestudy_api()

    # dispatch is constructed in `init_simulation_case_study`
    e = sim.dispatch({})
    e()
    a = e.results

    # construct the dispatch again with a different solver
    sim.solver = JaxSolver
    from diffrax import Dopri5
    sim.dispatch_constructor(diffrax_solver=Dopri5, rtol=1e-6)
    e = sim.dispatch({})
    e()
    b = e.results

    np.testing.assert_allclose(a.to_array(), b.to_array(), atol=1e-3)

    cpu_time_start = time.process_time()
    sim.benchmark(n=100)
    cpu_time_stop = time.process_time()

    t = cpu_time_stop - cpu_time_start

    if t > 1:
        raise AssertionError(f"Benchmarking took too long: {t}s. Expected t < 1s")


def test_rect_interpolation():
    sim: SimulationBase
    sim = init_bufferguts_casestudy(scenario="testing")

    # x input is defined on the interval [0,179]
    x_in = sim.parse_input(input="x_in", reference_data=sim.observations, drop_dims=[])
    # rect_interpolation adds duplicates the last y_179 for x_180
    x_in = rect_interpolation(x_in=x_in, x_dim="time")
    sim.model_parameters["x_in"] = x_in

    # Interpolations in diffrax now jump each discontinuity until the last time
    # that is evaluated by the solver. This is not jumped, so that it can be 
    # retrieved by SaveAt
    # https://github.com/patrick-kidger/diffrax/issues/58
    # It seems like this is the intended behavior of diffrax, 

    # run the simulation until exactly the last time, which is a discontinuity
    # this works and will not return a discontinuity, because jump_ts in the
    # PIDController of diffrax, is told that it shoud jump all ts that are 
    # smaller than x_stop
    sim.coordinates["time"] = np.linspace(0, 179, 1000)
    sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation produces no infinity values 
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0])
    )

    # test if the simulation also works with the normal time vector
    sim.reset_coordinate(dim="time")
    sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation produces no infinity values 
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0])
    )

    # run the simulaton until the added interpolation provided by rect_interpolation
    # until t=180
    sim.coordinates["time"] = np.linspace(0, 180, 1000)
    sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation works without problems if the last specified
    # value is not a discontinuity
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0])
    )

    # run the simulaton until the added interpolation provided by rect_interpolation
    # until t=180
    # This behavior is correctly caught, by pymob, before it can ocurr, because
    # an interpolation over the provided ts, and ys is not possible. And would
    # result in a difficult to diagnose max_steps error
    try:
        sim.coordinates["time"] = np.linspace(0, 180.01, 1000)
        sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    except AssertionError:
        threw_error = True

    if not threw_error: 
        AssertionError(
            "The solver should fail if interpolation is attempted "
            "to be done further than the intended range."        
        )

def test_no_interpolation():
    sim: SimulationBase
    sim = init_bufferguts_casestudy(scenario="testing")
    
    # x input is defined on the interval [0,179]
    x_in = sim.parse_input(input="x_in", reference_data=sim.observations, drop_dims=[])

    # run the simulaton until the provided x_input until t=179
    sim.coordinates["time"] = np.linspace(0, 179, 1000)
    sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation works without problems if the last specified
    # value is not a discontinuity
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0])
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_rect_interpolation()
    # test_benchmark_jaxsolver()
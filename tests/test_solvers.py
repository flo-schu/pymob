import os
import time
import numpy as np

from pymob.solvers.diffrax import JaxSolver
from tests.fixtures import init_simulation_casestudy_api


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


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_benchmark_jaxsolver()
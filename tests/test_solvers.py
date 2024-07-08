import os
import time
import pytest
import numpy as np
from click.testing import CliRunner
from pyinstrument import Profiler

from tests.fixtures import init_simulation_casestudy_api


def test_benchmark_time():
    sim = init_simulation_casestudy_api()

    cpu_time_start = time.process_time()
    sim.benchmark(n=100)
    cpu_time_stop = time.process_time()

    t = cpu_time_stop - cpu_time_start

    if t > 3:
        raise AssertionError(f"Benchmarking took too long: {t}s. Expected ~2s")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_commandline_API_infer()
import pytest
import numpy as np
from matplotlib import pyplot as plt

from pymob.simulation import SimulationBase

from tests.fixtures import init_test_case_study

def test_casestudy_api():
    sim = init_test_case_study("test_scenario")
    assert sim.model_parameter_dict == {'alpha': 0.5, 'beta': 0.02}





if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
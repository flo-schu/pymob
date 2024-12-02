import pytest
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from pymob.simulation import SimulationBase
from pymob.sim.config import Datastructure
from pymob.inference.scipy_backend import ScipyBackend

from tests.fixtures import (
    init_simulation_casestudy_api, 
    create_composite_priors,
    create_composite_priors_wrong_order,
)

def test_casestudy_api():
    sim = init_simulation_casestudy_api("test_scenario")
    assert sim.model_parameter_dict == {'alpha': 0.5, 'beta': 0.02}

def test_prior_parsing():
    params = create_composite_priors()
    parsed_params = ScipyBackend.parse_model_priors(
        parameters=params.free,
        dim_shapes={k:(100, 2) for k, _ in params.all.items()},
        data_structure=Datastructure()
    )

    inferer = ScipyBackend(SimulationBase())
    inferer.prior = parsed_params
    samples = inferer.sample_distribution()
    np.testing.assert_equal(samples["k"].shape, (100, 2))

def test_prior_parsing_error():
    params = create_composite_priors_wrong_order()
    try:
        parsed_params = ScipyBackend.parse_model_priors(
            parameters=params.free,
            dim_shapes={k:() for k, _ in params.all.items()},
            data_structure=Datastructure()
        )
        raise AssertionError("Parameter parsing should have failed.")
    except KeyError:
        pass


def test_prior_to_xarray():
    params = create_composite_priors()

    coords = {"experiment": (0, 1)}
    arr = params.k.to_xarray(coords)
    true = xr.DataArray([5,23.1], dims=("experiment",), coords={"experiment": [0,1]})

    assert np.all((arr == true).values)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
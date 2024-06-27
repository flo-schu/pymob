import pytest
import xarray as xr
import numpy as np
from click.testing import CliRunner

from pymob.simulation import SimulationBase

from fixtures import init_simulation_casestudy_api, linear_model

def test_simulation():
    sim = init_simulation_casestudy_api()

    evalu = sim.dispatch(theta=sim.model_parameter_dict)
    evalu()

    ds = evalu.results
    ds_ref = xr.load_dataset(f"{sim.data_path}/simulated_data.nc")

    np.testing.assert_allclose(
        (ds - ds_ref).to_array().values,
        0
    )

def test_minimal_simulation():
    sim = SimulationBase()
    linreg, x, y, y_noise, parameters = linear_model()

    obs = xr.DataArray(y_noise, coords={"x": x}).to_dataset(name="y")

    sim.config.simulation.dimensions = ["x"]
    sim.config.simulation.data_variables = ["y"]

    sim.observations = obs
    
    from pymob.sim.solvetools import solve_analytic_1d
    sim.model = linreg
    sim.solver = solve_analytic_1d

    # TODO: This should be implemented by default. 
    # Parameters need to be set as a copy
    sim.model_parameters["parameters"] = parameters.copy()
    sim.setup()
    evaluator = sim.dispatch(theta={"b":2})
    evaluator()

    evaluator.results

    sim.set_inferer("pyabc")
    sim.inferer.config.inference_pyabc.min_eps_diff = 0.001
    sim.inferer.run()
    sim.inferer


def test_indexing_simulation():
    pytest.skip()

def test_no_error_from_repeated_setup():
    sim = init_simulation_casestudy_api()  # already executes setup
    sim.setup()


def test_commandline_api_simulate():
    from pymob.simulate import main
    runner = CliRunner()
    
    args = "--case_study=test_case_study "\
        "--scenario=test_scenario"
    result = runner.invoke(main, args.split(" "))

    if result.exception is not None:
        raise result.exception


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    test_minimal_simulation()
    # test_scripting_API()
    # test_interactive_mode()

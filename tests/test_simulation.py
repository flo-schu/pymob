import pytest
import xarray as xr
import numpy as np
from click.testing import CliRunner

from pymob.simulation import SimulationBase
from pymob.sim.config import FloatParam

from tests.fixtures import init_simulation_casestudy_api, linear_model

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
    sim.observations = obs
    
    from pymob.sim.solvetools import solve_analytic_1d
    sim.model = linreg
    sim.solver = solve_analytic_1d

    sim.config.model_parameters.a = FloatParam(value=10, free=False)
    sim.config.model_parameters.b = FloatParam(value=3, free=True , prior="normal(loc=0,scale=10)")
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
    evaluator = sim.dispatch(theta={"b":3})
    evaluator()
    evaluator.results

    np.testing.assert_allclose(evaluator.results.y.values, y * 3 + 10)

    # this tests automatic updating of the parameterize method with partial
    sim.config.model_parameters.a = FloatParam(value=0, free=False)
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
    evaluator = sim.dispatch(theta={"b":3})
    evaluator()
    evaluator.results

    np.testing.assert_allclose(evaluator.results.y.values, y * 3)

    sim.config.model_parameters.sigma_y = FloatParam(free=True , prior="lognorm(scale=1,s=1)")
    sim.config.error_model.y = "normal(loc=y,scale=sigma_y)"

    sim.set_inferer("numpyro")
    sim.inferer.config.inference_numpyro.kernel = "nuts"
    # sim.inferer.config.inference_pyabc.min_eps_diff = 0.001
    sim.inferer.run()
    b = float(sim.inferer.idata.posterior["b"].mean()) # type: ignore
    sigma_y = float(sim.inferer.idata.posterior["sigma_y"].mean()) # type: ignore

    # test that the _model parameters of the Simulation remain unchanged. This is
    # achieved throgh deepcopying the dictionary on setting partial
    assert sim._model_parameters["parameters"]["b"] == 3
    np.testing.assert_allclose(b, parameters["b"], atol=0.05, rtol=0.05)
    np.testing.assert_allclose(sigma_y, parameters["sigma_y"], atol=0.05, rtol=0.05)



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

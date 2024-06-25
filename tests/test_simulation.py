import pytest
import xarray as xr
import numpy as np
from click.testing import CliRunner

from tests.fixtures import init_simulation_casestudy_api

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
    # test_scripting_API()
    # test_interactive_mode()

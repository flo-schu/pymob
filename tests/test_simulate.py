import pytest
import xarray as xr
import numpy as np
from click.testing import CliRunner

from tests.fixtures import init_test_case_study

def test_scripting_API():
    sim = init_test_case_study()
    sim.setup()

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
    sim = init_test_case_study()  # already executes setup
    sim.setup()


def test_interactive_mode():
    sim = init_test_case_study()
    sim.interactive()

def test_commandline_API():
    from pymob.simulate import main
    runner = CliRunner()
    
    args = "--case_study=test_case_study --scenario=test_scenario"
    result = runner.invoke(main, args.split(" "))


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_scripting_API()
    # test_interactive_mode()

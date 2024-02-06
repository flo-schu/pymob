import pytest
import xarray as xr
import numpy as np
from click.testing import CliRunner

from pymob.utils.store_file import prepare_casestudy

def load_test_case_study():
    config = prepare_casestudy(
        case_study=("test_case_study", "test_scenario"),
        config_file="settings.cfg"
    )
    from case_studies.test_case_study.sim import Simulation
    return Simulation(config=config)

def test_scripting_API():
    sim = load_test_case_study()
    evalu = sim.dispatch(theta=sim.model_parameter_dict)
    evalu()

    ds = evalu.results
    ds_ref = xr.load_dataset(f"{sim.data_path}/simulated_data.nc")

    np.testing.assert_allclose(
        (ds - ds_ref).to_array().values,
        0
    )

def test_indexing_simulation():
    pytest.fail()

def test_interactive_mode():
    sim = load_test_case_study()
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
    test_interactive_mode()
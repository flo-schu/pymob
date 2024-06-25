import json
import xarray as xr
import numpy as np
from click.testing import CliRunner
from matplotlib import pyplot as plt

from tests.fixtures import init_simulation_casestudy_api


def test_pymoo():
    sim = init_simulation_casestudy_api()
    sim.set_inferer(backend="pymoo")
    sim.inferer.run()

    with open(f"{sim.config.case_study.output_path}/pymoo_params.json", "r") as f:
        pymoo_results = json.load(f)

    estimated_parameters = pymoo_results["X"]
    true_parameters = sim.model_parameter_dict
    
    np.testing.assert_allclose(
        np.array(list(estimated_parameters.values())),
        np.array(list(true_parameters.values())),
        rtol=5e-2, atol=1e-5
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_scripting_api_pyabc()
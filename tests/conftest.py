# content of conftest.py
import numpy
import xarray
import pytest
from tests.fixtures import init_case_study_and_scenario

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["xr"] = xarray


# List test scenarios and simulations
@pytest.fixture(scope="session", params=[
    "test_scenario_v2",
    "lotka_volterra_hierarchical_presimulated_v1",
])
def scenario(request):
    return request.param

# List test backends
@pytest.fixture(scope="session", params=[
    "numpyro",
    # TODO: Add pymc once available
])
def backend(request):
    return request.param

# Derive simulations for testing from fixtures
# and run inference, so that the results from inference
# can be used for testing
@pytest.fixture(scope="session")
def sim_post_inference(scenario, backend):
    # TODO: This is enoguh once per session. This should be
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario=scenario, 
    )

    sim.set_inferer(backend=backend)
    if backend == "numpyro":
        sim.config.inference_numpyro.svi_iterations = 100
        sim.config.inference_numpyro.svi_learning_rate = 0.05

    sim.inferer.run()
    yield sim

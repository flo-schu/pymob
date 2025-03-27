import pytest
from pymob import SimulationBase
from pymob.sim.report import Report

###############################################
#        TEST TABLE PARAMETER ESTIMATES       #
###############################################

# define parameters

@pytest.fixture(params=["csv", "tsv", "latex"])
def parameter_estimates_format(request):
    return request.param

@pytest.fixture(params=[True, False])
def table_parameter_estimates_parameters_as_rows(request):
    return request.param

@pytest.fixture(params=[True, False])
def table_parameter_estimates_with_batch_dim_vars(request):
    return request.param

@pytest.fixture(params=["hdi", "sd"])
def parameter_estimates_error_metric(request):
    return request.param

@pytest.fixture(params=[{}, {"beta": "beta_test"}])
def table_parameter_estimates_override_names(request):
    return request.param

# define test function

def test_table_parameter_estimates(
    sim_post_inference: SimulationBase, 
    parameter_estimates_format,
    parameter_estimates_error_metric,
    table_parameter_estimates_parameters_as_rows,
    table_parameter_estimates_with_batch_dim_vars,
    table_parameter_estimates_override_names
):
    sim = sim_post_inference

    report = Report(config=sim.config, backend=type(sim.inferer))
    report.rc.debug_report = True
    report.rc.table_parameter_estimates_format = parameter_estimates_format
    report.rc.table_parameter_estimates_error_metric = parameter_estimates_error_metric
    report.rc.table_parameter_estimates_parameters_as_rows = table_parameter_estimates_parameters_as_rows
    report.rc.table_parameter_estimates_with_batch_dim_vars = table_parameter_estimates_with_batch_dim_vars
    report.rc.table_parameter_estimates_override_names = table_parameter_estimates_override_names
    
    _ = report.table_parameter_estimates(
        posterior=sim.inferer.idata.posterior,
        indices=sim.indices
    )

    assert report.status["table_parameter_estimates"]

###############################################
#      TEST POSTERIOR PREDICTIVE CHECKS       #
###############################################

# define test function

def test_posterior_predictive_checks(sim_post_inference: SimulationBase):
    sim_post_inference.posterior_predictive_checks()

import numpy as np
import xarray as xr
import pytest

from pymob.sim.config import Config, DataVariable, Modelparameters
from pymob.sim.parameters import Param, RandomVariable, Expression, OptionRV
from pymob.simulation import SimulationBase
from pymob.utils.store_file import prepare_casestudy

rng = np.random.default_rng(1)

def init_simulation_casestudy_api(scenario="test_scenario"):
    config = prepare_casestudy(
        case_study=("test_case_study", scenario),
        config_file="settings.cfg",
        pkg_dir="case_studies"
    )
    
    from case_studies.test_case_study.sim import Simulation
    sim = Simulation(config=config)
    sim.config.import_casestudy_modules(reset_path=True)
    sim.setup()
    return sim

def init_lotkavolterra_simulation_replicated():
    sim = init_simulation_casestudy_api()
    sim.config.case_study.scenario = "test_replicated"
    sim._observations = xr.Dataset()

    sim.config.simulation.batch_dimension = "id"

    sim.config.model_parameters.gamma = Param(value=0.3)
    sim.config.model_parameters.delta = Param(value=0.01)

    sim.config.data_structure.wolves = DataVariable(
        dimensions=["id", "time"], observed=False, dimensions_evaluator=["id", "time"])
    sim.config.data_structure.rabbits = DataVariable(
        dimensions=["id", "time"], observed=False, dimensions_evaluator=["id", "time"]
    )
    
    sim.coordinates["id"] = [0,1]
    sim.coordinates["time"] = [0,1,2,3,4,5]
    
    sim.config.simulation.y0 = ["wolves=Array([9, 5])", "rabbits=Array([40, 50])"]
    y0 = sim.parse_input("y0",drop_dims=["time"])
    sim.model_parameters["y0"] = y0
    
    sim.config.create_directory("results", force=True)
    sim.config.create_directory("scenario", force=True)
    sim.config.save(force=True)
    
    return sim


def init_bufferguts_casestudy(scenario="testing"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinator-tktd/case_studies"
    config.case_study.simulation = "BuffergutsSimulation"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()

def init_guts_casestudy_constant_exposure(scenario="testing_guts_constant_exposure"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinator-tktd/case_studies"
    config.case_study.simulation = "GutsSimulationConstantExposure"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()


def init_guts_casestudy_variable_exposure(scenario="testing_guts_variable_exposure"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinator-tktd/case_studies"
    config.case_study.simulation = "GutsSimulationVariableExposure"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()


def linear_model():
    def model(x, a, b):
        return a + x * b

    parameters = dict(
        a=0,
        b=1,
        sigma_y=1,
    )

    x = np.linspace(-5, 5, 50)
    y = model(x=x, a=parameters["a"], b=parameters["b"])
    y_noise = rng.normal(loc=y, scale=parameters["sigma_y"])

    return model, x, y, y_noise, parameters

def setup_solver(sim: SimulationBase, solver: type):
    sim.solver = solver
    sim.dispatch_constructor()
    return sim.evaluator._solver


def create_composite_priors():
    prior_mu = RandomVariable(distribution="normal", parameters={"loc": Expression("5"), "scale": Expression("2.0")})
    prior_k = RandomVariable(distribution="normal", parameters={"loc": Expression("mu"), "scale": Expression("1.0")})
    prior_k = RandomVariable(distribution="normal", parameters={"loc": Expression("mu + Array([5,5])"), "scale": Expression("1.0")})

    mu = Param(prior=prior_mu, hyper=True, dims=("experiment",))
    k = Param(prior=prior_k)

    theta = Modelparameters() # type: ignore
    theta.mu = mu
    theta.k = k

    return theta

def init_test_case_study_hierarchical() -> SimulationBase:
    config = Config()
    config.case_study.name = "test_case_study"
    config.case_study.scenario = "test_hierarchical"
    config.case_study.simulation = "HierarchicalSimulation"
    config.import_casestudy_modules(reset_path=True)
    Simulation = config.import_simulation_from_case_study()
    
    sim = Simulation(config)
    sim.setup()
    sim.config.save(force=True)

    return sim

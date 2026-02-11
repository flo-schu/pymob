import numpy as np
from tests.fixtures import init_lotka_volterra_UDE_case_study_from_settings

def test_convergence_optax_backend():
    sim = init_lotka_volterra_UDE_case_study_from_settings("InfererTest")

    sim.dispatch_constructor()

    sim.set_inferer("optax")

    sim.inferer.run()

    sim.model = sim.inferer.optimized_models[0]

    sim.dispatch_constructor()

    # Create an evaluator, run the simulation and obtain the results
    evaluator = sim.dispatch()
    evaluator()

    obs_prey = np.where(np.isnan(sim.observations.prey.values), evaluator.Y["prey"], sim.observations.prey.values)
    np.testing.assert_allclose(evaluator.Y["prey"], obs_prey, atol = 1, rtol = 1)

    obs_predator = np.where(np.isnan(sim.observations.predator.values), evaluator.Y["predator"], sim.observations.predator.values)
    np.testing.assert_allclose(evaluator.Y["predator"], obs_predator, atol = 1, rtol = 1)
import numpy as np
from tests.fixtures import (
    init_guts_casestudy_constant_exposure,
    init_guts_casestudy_variable_exposure,
)

# import jax with 64 bit precision
import jax
jax.config.update("jax_enable_x64", True)

def test_constant_exposure():
    sim = init_guts_casestudy_constant_exposure()

    sim.use_jax_solver()
    sim.dispatch_constructor(rtol=1e-15, atol=1e-16)
    evaluator = sim.dispatch(theta={})
    evaluator()
    
    sol_numerical = evaluator.results


    sim.use_symbolic_solver()
    evaluator = sim.dispatch(theta={})
    evaluator()

    sol_symbolic = evaluator.results

    diff = (
        sol_numerical.sel(time=[0, 180])
        - sol_symbolic.sel(time=[0, 180])
    )

    max_delta = np.abs(diff).max().to_array()
    np.testing.assert_array_less(max_delta, [1e-8, 1e-8, 1e-8])

    

def test_variable_exposure():
    sim = init_guts_casestudy_variable_exposure()

    sim.use_symbolic_solver()
    evaluator = sim.dispatch(theta={})
    evaluator()
    sol_symbolic = evaluator.results

    sim.use_jax_solver()
    sim.dispatch_constructor(rtol=1e-15, atol=1e-16)
    evaluator = sim.dispatch(theta={})
    evaluator()
    sol_numerical = evaluator.results

    # make sure errors are small between exact solution and numerical solution
    # the errors come from:
    # a) integrating not exactly to t_eq
    # b) using a numerical switch in the ODE solution.
    diff = (
        sol_numerical.sel(time=np.arange(0,sim.t_max))
        - sol_symbolic.sel(time=np.arange(0,sim.t_max))
    )[["D", "H", "S"]]
    max_delta = np.abs(diff).max().to_array()
    np.testing.assert_array_less(max_delta, [1e-8, 1e-8, 1e-8])

    axes = sim._plot.plot_multiexposure(sol_numerical, vars=["exposure", "D", "H", "S"], color="tab:blue", label_prefix="ODE")
    axes = sim._plot.plot_multiexposure(sol_symbolic, vars=["exposure", "D", "H", "S"], axes=axes, color="tab:red", linestyle="--", label_prefix="exact")
    fig = axes[0].figure
    fig.savefig(f"{sim.output_path}/solution_comparison.png")




if __name__ == "__main__":
    test_constant_exposure()
    test_variable_exposure()
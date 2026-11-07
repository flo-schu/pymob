# Bugs

## Open

- **2026-11-06** – `SimulationBase.parse_input` (`pymob/simulation.py:1105`) raises
  `TypeError: only 0-dimensional arrays can be converted to Python scalars` (likely numpy 2 behavior).
  Found via guts_base `tests/test_scripted_simulations.py` (2 tests). Not fixed (out of scope).

## Closed

- **2026-11-06** – ODE models written for pymob < 0.7 failed with the JaxSolver after commit `9a74a48`
  (`ValueError: Cannot broadcast to shape with fewer dimensions: arr_shape=(1,) shape=()`).
  Fixed on branch `fix/ode-output-shape-compat` (`SolverBase._match_state_shapes`).
  Details: `2026-11-06_2326_ode-output-shape-compat/summary.md`.

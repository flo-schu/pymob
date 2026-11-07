# ODE output shape compatibility (JaxSolver)

Origin: failing guts_base test `tests/test_io.py::test_copy[GutsBase-red_sd]`.
Full diagnosis, plan (revisions 1 and 2), scratch scripts and raw test results are in the guts_base repo:
`../guts_base/ai-logs/2026-11-06_1515_fix-test-copy/` (`summary.md`, `plan.md`).

## Problem
Commit `9a74a48` ("Allow zero-dimensional solver input") stopped promoting scalar ODE states and parameters
to 1-D arrays of length 1. Models written before that commit return derivatives of shape `(1,)` for
states of shape `()`, so diffrax fails with
`ValueError: Cannot broadcast to shape with fewer dimensions: arr_shape=(1,) shape=()`.

## User decisions
- Do not change guts_base models (a fixed exposure input with an extra dimension is a common pattern there).
- Make pymob backwards-compatible and still support scalar states: option A (shape adapter).
- Warn (`DeprecationWarning`) when reshaping.
- The check belongs in `SolverBase` (user feedback during implementation).
- Unrelated baseline bugs stay out of scope, but are recorded in the right repo.

## Implementation (branch `fix/ode-output-shape-compat`)
- `SolverBase._match_state_shapes(f, y_shapes, out_shapes, num_backend)` in `pymob/solvers/base.py`:
  - equal shapes → `f` unchanged (no overhead, no warning)
  - shapes differ only by axes of length 1 → returns a wrapped `f` that reshapes the derivatives, with a `DeprecationWarning`
  - other mismatch → `PymobError` (sections: brief message, why, how to fix, details)
- `JaxSolver.odesolve` (`pymob/solvers/diffrax.py`) computes the output shapes with `jax.eval_shape`
  (static, once at trace time) and calls the helper.
- Tests in `tests/test_solvers.py`: `test_solver_scalar_states_without_warning`,
  `test_solver_legacy_model_output_shapes`, `test_solver_model_output_shape_mismatch_error`.
  I checked that the 2 behavior tests fail without the change (git stash).
- `CHANGELOG.md` created (none existed), `[Unreleased]` entry.

## Design decisions
- The reshape is allowed only if the shapes are equal after removing all length-1 axes. The number of values never changes.
- The output shapes are computed by the backend (jax) and passed to `SolverBase`. This keeps `SolverBase`
  backend-agnostic (`num_backend` parameter, like `preprocess_*`).
- The scipy solver is not affected (it works on flattened states).

## Tests (raw results in the guts_base ai-logs folder)
- pymob fast: baseline 177 passed / 21 skipped → 180 passed / 21 skipped (3 new tests).
- pymob has no slow-marked tests.
- guts_base fast: the 7 broadcast failures are fixed (13 failed + 24 errors → 6 failed + 24 errors; the rest are unrelated bugs).
- guts_base slow subset (`test_lp50`, `test_inference_from_config[numpyro]` for red_sd and red_sd_da): 4 passed.

## Bugs found
See `../bugs.md` (pymob) and `../guts_base/ai-logs/bugs.md`.

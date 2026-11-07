# Tasks

## Open

- **2026-11-06** – Remove the deprecated automatic reshaping of ODE derivatives in
  `SolverBase._match_state_shapes` in a future major version (after models were updated).

## Closed

- **2026-11-06** – Explicit error for mismatching ODE model output shapes; backwards-compatible
  reshaping for models written for pymob < 0.7 (branch `fix/ode-output-shape-compat`).

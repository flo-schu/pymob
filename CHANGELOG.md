# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- ODE models written for pymob < 0.7 work again with scalar ODE states. These models return
  derivatives with extra axes of length 1 (e.g. shape `(1,)` for a state of shape `()`).
  The `JaxSolver` now reshapes such derivatives to the shapes of the ODE states
  (`SolverBase._match_state_shapes`).

### Added

- A `PymobError` with an explanation is raised if the shapes of the derivatives returned by an
  ODE model do not match the shapes of the ODE states. Before, diffrax raised an unclear broadcasting error.

### Deprecated

- Automatic reshaping of derivatives that differ from the ODE state shapes only by axes of length 1.
  A `DeprecationWarning` is issued. Return derivatives with the same shape as the ODE states.

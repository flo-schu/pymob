# Diagnosing problems in models

For structural problems `pymob` should offer constructive error messages, before
problems occurr. Still during parameter inference and optimization many problems
can ocurr, the below is an unsorted list of typical problems and ways to diagnose
and fix such problems.

## NaN likelihoods with `JaxSolver`

+ First make sure this problem is not related to the choice and the parameterization
  of the solver. In the {meth}`pymob.simulation.SimulationBase.dispatch_constructor` Try increasing `max_steps`, 













## Further reference


# Case studies

Case studies are a principled approach to the *modelling* process. In essence, they are a simple template that contains building blocks for model and names and stores them in an intuitive and reproducible way.

Each case study consists of the following components:

```
test_case_study
  sim.py               # sets up the simulation
  mod.py               # (opt) outsources solver and model definitions
  data.py              # (opt) outsources data input
  plot.py              # (opt) outsources vizualizations

  scenarios
    scenario_A         # the scenario of the "test_case_study" 
      settings.cfg     # configuration for the case study and scenario
      simulation.cfg   # parameters of the simulation

  results
    scenario_A         # the results of "scenario_A"

  data                 # (optional) datafiles for data.py

  scripts              # (optional) evaluation scripts

  docs                 # (optional) documentation of the case study
```

While it is recommended to keep data, docs, results, scripts directories in each case study to keep a comprehensive and compact structure of the project, these can reside anywhere else. 

## Configuration


Settings files are created as conf `.cfg` files. These files are organized into the following sections.

```{admonition}
:class: attention
This will change in the next version of `pymob` configuration is then done in the scripting API, but config files can still be exported and imported
from the Simulation instance. This will make configuration considerably more user friendly as the possible options are directly availble through type hints.
```

```conf
[case-study]
output = 
data = 
simulation = 

[simulation]
# model specification
# --------------------
model = 
solver = 
solver_post_processing = 
y0 = 
seed = 

# data description
# --------------------
dimensions = time
evaluator_dim_order = time
data_variables = 
data_variables_max = 
data_variables_min = 

[free-model-parameters]
...

[fixed-model-parameters]
...

[error-model]
...

[inference]
...

[inference.pyabc]
...

[inference.pymoo]
...

[inference.numpyro]
...

```

### case-study configuration

Contains the information about the Simulation class to be used and the data and output directory. The paths are relative to the root directory (where the case study is launched)

### Simulation configuration

Contains the details about the simulation. If an ode model is used, it should be specified here. In that case also different solvers can be provided to the simulation. 

It is also possible to provide the solver directly which then executes the entire simulation. In this case the solver should be a wrapper around a possibly external simulation which returns the results as a dictionary with keys, corresponding to the data variables.

**Solver post processing** can be used if the results returned from the solver need some additional post processing before they can be compared with the observations. The method takes a dictionary as an input and potentially other arguments specified e.g. in the free-model-parameters or fixed-model-parameters. 

**y0** Initial values of the ODE model. Can be any list separated by whitespace that follows `sympy` syntax. E.g. `wolves=Array([2]) rabbits=rabbits`. This can be processed with `Simulation.parse_input(data=[observations], input="y0", drop_dims="time")`. This will create an array of the keys (wolves, rabbits) and broadcast the values along all coordinates of the observations, but retaining only the first value of the dropped dimensions if a variable (rabbits) was provided. This is very useful if some data_variables of the ODE model were observed and the initial value should be taken from the data.

**seed**. The seed is used to initate random processes for reproducibility. The behavior is still experimental.

### free-model-parameters

Model parameters that are subject to parameter inference and can be varied in the parameter estimation process or in the interactive simulation 

### fixed-model-parameters

Model parmameters that remain fixed throughout the simulation

### error model

Error functions for comparing the simulation results to the data. These functions will also be parsed with `sympy` parsers. Still experimental

### inference

Options for adjusting the inference algorithms
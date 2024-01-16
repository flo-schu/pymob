# Case studies

Case studies are a principled approach to the *modelling* process. In essence, they are a simple template that contains building blocks for model and names and stores them in an intuitive and reproducible way.

Each case study consists of the following components:

```
test_case_study
  mod.py               # the model
  sim.py               # the solver (forward simulation)
  data.py              # (opt) data input
  plot.py              # (opt) vizualizations

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


### Case study settings (settings.cfg)

Settings files are created as conf `.cfg` files. These files are organized into sections. E.g.:

```conf
[case-study]
output = .

[simulation]

```


### Simulation configuration files (config.json)

Configuration options for the simulations are given as JSON files. The reason for this is that json files allow the possibility of nesting, which makes it easy to parameterize a simulation on different hierarchy levels.

If a simulator requires a different filetype of parameter input, an adapter needs to be written, that translates the parameters specified in the json files to the format required by the simulator.

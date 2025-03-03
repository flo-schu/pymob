# Interactive simulation of test case study

First load packages and switch into the correct working directory


```python
from pymob import Config

from lotka_volterra_case_study.sim import Simulation_v2
```

Load casestudy


```python
config = Config("../scenarios/test_scenario_v2/settings.cfg")
config.case_study.package = "../.."

sim = Simulation_v2(config)
sim.setup()

```

    MinMaxScaler(variable=rabbits, min=5.968110437683305, max=86.99133665713266)
    MinMaxScaler(variable=wolves, min=7.203778019337644, max=62.829641338400535)
    Results directory exists at '/home/flo-schu/projects/pymob/case_studies/lotka_volterra_case_study/results/test_scenario_v2'.
    Scenario directory exists at '/home/flo-schu/projects/pymob/case_studies/lotka_volterra_case_study/scenarios/test_scenario_v2'.


    /home/flo-schu/miniconda3/envs/lotka-volterra/lib/python3.11/site-packages/pymob/simulation.py:546: UserWarning: The number of ODE states was not specified in the config file [simulation] > 'n_ode_states = <n>'. Extracted the return arguments ['dprey_dt', 'dpredator_dt'] from the source code. Setting 'n_ode_states=2.
      warnings.warn(



```python
# Prey birth rate (alpha * prey)
sim.config.model_parameters.alpha.min = 0.1
sim.config.model_parameters.alpha.max = 1.0
sim.config.model_parameters.alpha.free = True

# Predation rate (- beta * prey * predator)
sim.config.model_parameters.beta.min = 0.005
sim.config.model_parameters.beta.max = 0.05
sim.config.model_parameters.beta.free = True

# Predator reproduction rate (delta * prey * predator)
sim.config.model_parameters.delta.min = 0.005
sim.config.model_parameters.delta.max = 0.05
sim.config.model_parameters.delta.free = True

# Predator death rate (- gamma * predator)
sim.config.model_parameters.gamma.min = 0.1
sim.config.model_parameters.gamma.max = 1.0
sim.config.model_parameters.gamma.free = True

```

## Run interactive simulation


```python
sim.interactive()
```


    HBox(children=(VBox(children=(FloatSlider(value=0.5, description='alpha', max=1.0, min=0.1, step=None), FloatS…


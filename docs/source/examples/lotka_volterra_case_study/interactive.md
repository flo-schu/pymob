# Interactive simulation of test case study

First load packages and switch into the correct working directory


```python
import os

import numpy as np
import matplotlib.pyplot as plt
from pymob.utils.store_file import prepare_casestudy

os.chdir("../../../")
```

Load casestudy


```python

config = prepare_casestudy(
    case_study=("lotka_volterra_case_study", "test_scenario"),
    config_file="settings.cfg",
    pkg_dir="case_studies"
)
from case_studies.lotka_volterra_case_study.sim import Simulation
```

## Run interactive simulation


```python
sim = Simulation(config=config)
sim.interactive()
```

    MinMaxScaler(variable=rabbits, min=0.0, max=86.99133666713266)
    MinMaxScaler(variable=wolves, min=0.0, max=62.829641348400536)



    HBox(children=(VBox(children=(FloatSlider(value=0.5, description='alpha', max=1.0, step=0.01), FloatSlider(val…


# Pymob quickstart

## Initialize a simulation

In pymob a Simulation object is initialized by calling the `SimulationBase` class from the simulation module.

```py
from pymob.simulation import SimulationBase

sim = SimulationBase()
```

```{admonition} Configuring the simulation
:class: hint
Optionally, we can configure the simulation at this stage with 
`sim.config.case_study.name = "linear-regression"`, `sim.config.case_study.scenario = "test"`, and many more options. 
```

## Define a model 

Let's investigate a linear regression as the most simple task.

```py
def linreg(x, a, b):
    return a + x * b
```

So we assume that this model describes our data well. So we add it to the simulation

```py
sim.model = linreg
```

## Defining a solver

In our case the model gives the exact solution of the model.
Solvers in pymob are callables that need to return a dictionary of results mapped to the data variables

```py
from pymob.sim.solvetools import solve_analytic_1d
sim.solver = solve_analytic_1d
```

## Generate artificial data

In the real world, you will have measured a dataset. For demonstration, we define parameters $theta$, that we assume describe the true data generating process and generate observations $y$. Then we
generate data for $x$ on [-5, 5] and add random noise with a standard deviation of $\sigma_y = 1$.

```py
import numpy as np
from scipy.stats import norm

# define the coordinates of the x-dimension to generate data for
x = np.linspace(-5, 5, 50)

# define a set of parameters θ
theta = dict(a=0, b=1, sigma_y=1)

# then simulate some data and add some noise
y = linreg(x=x, a=theta["a"], b=theta["b"])
y_noise = norm(loc=y, scale=theta["sigma_y"]).rvs()
```

## The pymob magic 🪄

So far we have not done anythin special. Pymob exists, because wrangling dimensions of input and output data, nested data-structures, missing data is painful. We avoid most of the mess by using `xarray` as a common input/output format. So we have to transform our data into a `xarray.Dataset` and add it to the simulation.

```py
import xarray as xr

obs = xr.DataArray(y_noise, coords={"x": x}).to_dataset(name="y")
```

Next we have to let `pymob` know about the data structure of our observations. We add it to the config module. Of course more dimensions and data-variables can be provided, but in our simple example, we have the dimension `x` and the output variable `y`.

```py
sim.config.simulation.dimensions = ["x"]
sim.config.simulation.data_variables = ["y"]

sim.observations = obs
```

This worked 🎉 

```{admonition} Scalers
:class: hint
We also notice a mysterious Scaler message. This tells us that our data variable has been identified and a scaler was constructed, which transforms the variable between [0, 1]. This has no effect at the moment, but it can be used later. Scaling can be powerful to help parameter estimation in more complex models.
```

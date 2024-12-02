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
rng = np.random.default_rng(1)

# define the coordinates of the x-dimension to generate data for
x = np.linspace(-5, 5, 50)

# define a set of parameters θ
theta = dict(a=0, b=1, sigma_y=1)

# then simulate some data and add some noise
y = linreg(x=x, a=theta["a"], b=theta["b"])
y_noise = rng.normal(loc=y, scale=parameters["sigma_y"])
```

## The pymob magic 🪄

So far we have not done anythin special. Pymob exists, because wrangling dimensions of input and output data, nested data-structures, missing data is painful. We avoid most of the mess by using `xarray` as a common input/output format. So we have to transform our data into a `xarray.Dataset` and add it to the simulation.

```py
import xarray as xr

sim.observations = xr.DataArray(y_noise, coords={"x": x}).to_dataset(name="y")
```

This worked 🎉 `sim.config.data_structure` will now give us some information about the layout of our data, which will handle the data transformations in the background.

We can give `pymob` additional information about the data structure of our observations and intermediate (unobserved) variables that are simulated. This can be done with `sim.config.data_structure.y = DataVariable(dimensions=["x"])`.
These information can be used to switch the dimensional order of the observations or provide data variables that have differing dimensions from the observations, if needed. But if the dataset is ordinary, simply setting `sim.observations` property with a `xr.Dataset` will be sufficient.

```{admonition} Scalers
:class: hint
We also notice a mysterious Scaler message. This tells us that our data variable has been identified and a scaler was constructed, which transforms the variable between [0, 1]. This has no effect at the moment, but it can be used later. Scaling can be powerful to help parameter estimation in more complex models.
```

## Parameterizing a model

Parameters are specified via the `FloatParam` or `ArrayParam` class. Parameters can be marked free or fixed depending on whether they should be variable during an optimization procedure.

```py
from pymob.sim.config import FloatParam
sim.config.model_parameters.a = FloatParam(value=10, free=False)
sim.config.model_parameters.b = FloatParam(value=3, free=True)
# this makes sure the model parameters are available to the model.
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
```

`sim.model_parameters` is a dictionary that holds the model input data. The keys it takes by default are `parameters`, `y0` and `x_in`. In our case, we have a analytic model and need only `parameters`. In situations, where initial values for variables are needed, they can be provided with `sim.model_parameters["y0"] = ...`. 

```{admonition} generating input for solvers
:class: hint
A helpful function to generate `y0` or `x_in` from observations is `SimulationBase.parse_input`, combined with settings of `config.simulation.y0`
```

## Running the model 🏃

The model is prepared with a parameter set and executed with 

```py
evaluator = sim.dispatch(theta={"b":3})
evaluator()
evaluator.results
```

This returns a dataset which is of the exact same shape as the observation dataset, plus intermediate variables that were created during the simulation, if they are tracked by the solver.

Although this API seems to be a bit clunky, it is necessary, to make sure that simulations that are executed in parallel are isolated from each other.


## Estimating parameters 

We are almost set infer the parameters of the model. We add another parameter to also estimate the error of the parameters, We use a lognormal distribution for it. We also specify an error model for the distribution. This will be 

$$y_{obs} \sim Normal (y, \sigma_y)$$

```py
sim.config.model_parameters.sigma_y = FloatParam(free=True , prior="lognorm(scale=1,s=1)")
sim.config.error_model.y = "normal(loc=y,scale=sigma_y)"
```

```{admonition} numpyro distributions
:class: warning
Currently only few distributions are implemented in the numpyro backend. This API will soon change, so that basically any distribution can be used to specifcy parameters. 
```

Finally, we let our inferer run the paramter estimation procedure with the numpyro backend and a NUTS kernel. This does the job in a few seconds

```py
sim.set_inferer("numpyro")
sim.inferer.config.inference_numpyro.kernel = "nuts"
sim.inferer.run()

sim.inferer.idata.posterior
```

We can inspect our estimates and see that the parameters are well esimtated by the model.

## Exporting the simulation and running it via the case study API

After constructing the simulation, all settings of the simulation can be exported to a comprehensive configuration file, along with all the default settings. This is as simple as 

```py
sim.config.case_study.name = "quickstart"
sim.config.case_study.scenario = "test"
sim.config.create_directory("scenario", force=True)
sim.config.create_directory("results", force=True)
sim.save_observations(force=True)
sim.config.save(force=True)
```

The simulation will be saved to the default path (`CASE_STUDY/scenarios/SCENARIO/settings.cfg`) or to a custom path spcified with the `fp` keyword. `force=True` will overwrite any existing config file, which is the reasonable choice in most cases.

From there on, the simulation is (almost) ready to be executable from the commandline.

### Commandline API

The commandline API runs a series of commands that load the case study, execute the {meth}`pymob.simulation.SimulationBase.initialize` method and perform some more initialization tasks, before running the required job.

+ `pymob-infer`: Runs an inference job e.g. `pymob-infer --case_study=quickstart --scenario=test --inference_backend=numpyro`. While there are more commandline options, these are the two required 
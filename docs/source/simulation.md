# Simulation

The modelling toolkit offers a variety of algorithms that can be used on a large number of models. Reckognizing that simulation, optimization/calibration, parameter inference, sensitivity analysis, validation, etc. require similar workflows and have common input and output datastreams opens the door for building a generic Simulation class, which can be reused for the listed purposes.

## Simulation components

Any simulation has recurring components. Facilitating transfer of information
between those components is key to analyze a described model with different 
tools. In this framework, simulations are defined as classes which require
the definition of methods that define the simulation.

```python
from timepath import SimulationBase

class Simulation(SimulationBase):
    def parameterize(self, input):
        # Initial conditions and parameters

    def set_coordinates(self, input):
        # the model is integrated over the dimension time

    def observations(self, input):
        # (optional) Defines the observations of the model
        # Needed for calibration, inference and validation tasks

    def run(self):
        # describe the model, solve it and return the solution

```


### The model

This is defined by the `run` method. It returns the output of a single simulation
run and takes no arguments. Extra input to the function is specified in the
`initialize` method and can then be accessed as a class attribute.

```python
def run(self):
    y0 = self.y0
    params = self.model_parameters
    t = self.coordinates["time"]

    model = ...  # describe the model

    solution = ...  # Solve the model
    return solution
```

The solution is returned as a `list` or `np.array`. Importantly, the shape of
the output must follow the order of the dimensions returned in `set_coordinates`.

### Model parameters

Model parameters are a critical part of the model. Changing model parameters
changes the output of a model. Parameters are involved in many typical tasks
of model development:

- parameter calibration (or optimization)
- parameter inference (slightly different operation)
- senstivity analysis

In addition to the model parameters, the `parameterize` method also returns 
the initial conditions of the simulation. 

```python
def parameterize(self, input):
    # Initial conditions and parameters
    y0 = ...
    parameters = ...
    return y0, parameters
```

The input argument contains a list of file-paths. If any files were provided
in the simulation `settings.cfg` configuration file. These files can then be 
read and parameter names and values imported. This is particularly useful if
parameters are provided in parallelization environments on computing clusters.
For storing parameters, the use of `JSON` files is recommended. Those can be
directly parsed as python dictionaries, which is the preferred format for 
parameters, since they can be easily forwarded as keyword arguments in 
functions. For this, a convenience function is included.

```json
{
    "y0": {
        "salad": 10, 
        "rabbits": 2},
    "parameters": {
        "eating_speed": 0.2,
        "growth_rate": 0.3,
    }
}
```

```python
from timepath.store_file import read_config
def parameterize(self, input):
    parameters = read_config(input[0])
    y0 = parameters["y0"]
    params = parameters["parameters"]
    return y0, parameters
```


### Simulation coordinates

The model coordinates describe the dimensions over which the model is solved.
In a simple model only one coordinate is present. This dimension can be the
time dimension.

A very simple example:

```python
def set_coordinates(self, input)
    time = np.linspace(0, 100, 50)
    return time
```

Models can also have a batch dimension, if it is part of the model that multiple
replicates are simulated as one trial of an experiment.

```python
def set_coordinates(self, input)
    time = np.linspace(0, 100, 50)
    sample = [1,2,3]
    return time, sample
```

Any number of dimension can be specified as long as those dimensions are also
named in the same order in the `settings.cfg` file

```conf
[simulation]
dimensions = time sample
```
### Observations

Observations must have the same form as the model output

## Input / Output 

As a uniform exchange format `netcdf` is enforced by the package. In python `netcdf` files are handled by the `xarray` package. For convenience, data_variables of interest can be saved as a number of output formats. However, higher dimesnional datasets with more coordinates than e.g. time, must be aggregated over the remaining dimensions, before they can be processed to .csv files

TODO: make sure this works
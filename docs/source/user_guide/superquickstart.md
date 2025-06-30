# Pymob in minutes - the basics

This guide provides a streamlined introduction to the basic Pymob workflow and its key functionalities.  
We will explore a simple linear regression model that we want to fit to a noisy dataset.  
Pymob supports the modeling process by providing several tools for *data structuring*, *parameter estimation* and *visualization of results*.  
  
If you are looking for a more detailed introduction, [click here](https://pymob.readthedocs.io/en/stable/user_guide/introduction.html).  
If you want to learn how to work with ODE models, check out [this tutorial](). 

## Pymob components 🧩

Before starting the modeling process, let's take a look at the main steps and modules of pymob:

1. __Simulation:__   
First, we need to initialize a Simulation object by creating an instance of the {class}`pymob.simulation.SimulationBase` class from the simulation module.   
Optionally, we can configure the simulation with `sim.config.case_study.name = "linear-regression"`, `sim.config.case_study.scenario = "test"` and many other options. 

2. __Model:__   
Our model will be defined as a standard python function.  
We will then assign it to the Simulation object by accessing the `.model` attribute. 

3. __Observations:__   
Our observation data must be structured as an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).  
We assign it to the {attr}`~pymob.sim.config.Casestudy.observations` attribute of our Simulation object.   
Calling `sim.config.data_structure` will give us further information about the layout of our data.  

4. __Solver:__  
A [solver](https://pymob.readthedocs.io/en/stable/api/pymob.solvers.html) is required to solve the model.   
In our simple case, we will use the `solve_analytic_1d` solver from the {mod}`~pymob.solver.analytic` module.  
We assign it to our Simulation object using the {attr}`~pymob.simulation.solver` attribute.   
Since our model already provides an analytical solution, this solver basically does nothing. It is still needed to fulfill Pymob's requirement for a solver component.   
For more complex models (e.g. ODEs), the `JaxSolver` from the {mod}`~pymob.solver.diffrax` module is a more powerful option.   
Users can also implement custom solvers as a subclass of {class}`pymob.solver.SolverBase`.   
  
5. __Inferer:__  
The inferer handels the parameter estimation.  
Pymob supports [various backends](https://pymob.readthedocs.io/en/stable/user_guide/framework_overview.html). In this example, we will work with *NumPyro*.  
We assign the inferer to our Simulation object via the {attr}`~pymob.simulation.inferer` attribute and configure the desired kernel (e.g. *nuts*).  
But before inference, we need to parameterize our model using the *Param* class.   
Each parameter can be marked either as free or fixed, depending on whether it should be variable during the optimization procedure.   
The parameters are stored in the {attr}`~pymob.simulation.SimulationBase.model_parameters` dictionary, which holds model input values.
By default, it takes the keys: `parameters`, `y0` and `x_in`. 

6. __Evaluator:__  
The Evaluator is an instance to manage model evaluations. It sets up tasks, coordinates parallel runs of the simulation and keeps track of the results from each simulation or parameter inference process.   
Evaluators store the raw output from a simulation and can generate an xarray object from it that corresponds to the data-structure of the observations with the {attr}`~pymob.sim.evaluator.Evaluator.results` property. This automatically aligns the simulations results with the observations, for simple computation of loss functions.  

7. __Config:__  
The simulation settings will be saved in a `.cfg` configuration file.  
The config file contains information about our simulation in various sections. [Learn more here](https://pymob.readthedocs.io/en/stable/user_guide/case_studies.html#configuration).  
We can further use it to create new simulations by loading settings from a config file. 

![framework-overview](.\figures\pymob_overview.png)

## Getting started 🛫


```python
# First, import the necessary python packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Import the pymob modules
from pymob.simulation import SimulationBase
from pymob.sim.solvetools import solve_analytic_1d
from pymob.sim.config import Param
```

Since no measured data is provided, we will generate an artificial dataset.  
$y_{obs}$ represents the **observed data** over the time $t$ [0, 10].  
To use this data later in the simulation, we need to convert it into an **xarray-Dataset**.  
In your own application, you would replace this with your measured experimental data.  


```python
# Parameter for the artificial data generation
rng = np.random.default_rng(seed=1)  # for reproducibility
slope = rng.uniform(2,4)
intercept = 1.0
num_points = 100
noise_level = 1.7

# generating time values
t = np.linspace(0, 10, num_points)

# generating y-values with noise
noise = np.random.normal(0, noise_level, num_points)
y_obs = slope * t + intercept + noise

# visualizing our data
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(t, y_obs, label='Datapoints')
ax.set(xlabel='t [-]', ylabel='y_obs [-]', title ='Artificial Data')
plt.tight_layout()

# convert the data to an xr-Dataset
data_obs = xr.DataArray(y_obs, coords={"t": t}).to_dataset(name="y")
data_obs
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base rgba(0, 0, 0, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, white)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
  );
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base, rgba(255, 255, 255, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, #111111)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
  );
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
  border: 2px solid transparent !important;
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0) !important;
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  border-color: var(--xr-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
  border-color: var(--xr-background-color-row-even);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  border-top: 2px dotted var(--xr-background-color);
  padding-bottom: 20px !important;
  padding-top: 10px !important;
}

.xr-var-attrs-in + label,
.xr-var-data-in + label,
.xr-index-data-in + label {
  padding: 0 1px;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-data > pre,
.xr-index-data > pre,
.xr-var-data > table > tbody > tr {
  background-color: transparent !important;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}

.xr-var-attrs-in:checked + label > .xr-icon-file-text2,
.xr-var-data-in:checked + label > .xr-icon-database,
.xr-index-data-in:checked + label > .xr-icon-database {
  color: var(--xr-font-color0);
  filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
  stroke-width: 0.8px;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 2kB
Dimensions:  (t: 100)
Coordinates:
  * t        (t) float64 800B 0.0 0.101 0.202 0.303 ... 9.697 9.798 9.899 10.0
Data variables:
    y        (t) float64 800B 1.59 2.136 1.343 2.532 ... 31.06 27.02 30.87 32.09</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-b6e66973-2041-42d5-8705-ecf533c55a89' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b6e66973-2041-42d5-8705-ecf533c55a89' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>t</span>: 100</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-77264d36-2121-47a0-9814-7048a5037bab' class='xr-section-summary-in' type='checkbox'  checked><label for='section-77264d36-2121-47a0-9814-7048a5037bab' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>t</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.101 0.202 ... 9.899 10.0</div><input id='attrs-914d5174-447d-4c2c-b6c2-2cd02c911b23' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-914d5174-447d-4c2c-b6c2-2cd02c911b23' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c0677693-e243-4c3d-bd9f-6ceef5966d2f' class='xr-var-data-in' type='checkbox'><label for='data-c0677693-e243-4c3d-bd9f-6ceef5966d2f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.      ,  0.10101 ,  0.20202 ,  0.30303 ,  0.40404 ,  0.505051,
        0.606061,  0.707071,  0.808081,  0.909091,  1.010101,  1.111111,
        1.212121,  1.313131,  1.414141,  1.515152,  1.616162,  1.717172,
        1.818182,  1.919192,  2.020202,  2.121212,  2.222222,  2.323232,
        2.424242,  2.525253,  2.626263,  2.727273,  2.828283,  2.929293,
        3.030303,  3.131313,  3.232323,  3.333333,  3.434343,  3.535354,
        3.636364,  3.737374,  3.838384,  3.939394,  4.040404,  4.141414,
        4.242424,  4.343434,  4.444444,  4.545455,  4.646465,  4.747475,
        4.848485,  4.949495,  5.050505,  5.151515,  5.252525,  5.353535,
        5.454545,  5.555556,  5.656566,  5.757576,  5.858586,  5.959596,
        6.060606,  6.161616,  6.262626,  6.363636,  6.464646,  6.565657,
        6.666667,  6.767677,  6.868687,  6.969697,  7.070707,  7.171717,
        7.272727,  7.373737,  7.474747,  7.575758,  7.676768,  7.777778,
        7.878788,  7.979798,  8.080808,  8.181818,  8.282828,  8.383838,
        8.484848,  8.585859,  8.686869,  8.787879,  8.888889,  8.989899,
        9.090909,  9.191919,  9.292929,  9.393939,  9.494949,  9.59596 ,
        9.69697 ,  9.79798 ,  9.89899 , 10.      ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-80914dc8-dc91-498c-b781-0761b0f62517' class='xr-section-summary-in' type='checkbox'  checked><label for='section-80914dc8-dc91-498c-b781-0761b0f62517' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.59 2.136 1.343 ... 30.87 32.09</div><input id='attrs-0c997349-0f4e-4053-836f-48a9cad4415e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0c997349-0f4e-4053-836f-48a9cad4415e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-86a643cf-ad83-418a-b5e3-5f52ca258650' class='xr-var-data-in' type='checkbox'><label for='data-86a643cf-ad83-418a-b5e3-5f52ca258650' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 1.58976396,  2.13626768,  1.34289925,  2.53150209,  2.56102578,
       -0.21899969,  1.26689629,  2.064798  ,  4.04579132,  2.33368407,
        5.56713047,  4.64800357,  4.12075281,  2.8115269 ,  6.05279673,
        4.72113173,  9.74281027,  6.23928579,  6.02924706,  8.057377  ,
        8.67482637,  6.44336997,  9.71115216,  6.84397881,  9.06017053,
        6.49025613, 10.12391111,  9.5022547 ,  8.05685756,  9.53097276,
        8.43057554, 14.57156966,  8.77743968,  8.75197054, 11.73207016,
       10.08294682, 13.55943288, 14.76145888, 14.13237262, 15.826024  ,
       11.69458685, 12.51989808, 14.40128584, 14.39427702, 12.66837215,
       15.92469856, 17.83174566, 17.44936835, 17.64962036, 14.75949057,
       15.63025287, 17.29111458, 19.47794167, 16.11896414, 19.22733081,
       15.55895925, 17.50982302, 16.59275063, 19.37052338, 18.53681002,
       21.55941223, 19.05813279, 18.82898749, 18.51376136, 19.01012364,
       20.79403644, 22.02425154, 21.93984824, 21.0715503 , 20.06227644,
       22.79902669, 20.19672578, 24.33260566, 25.66898506, 22.59631756,
       24.35184169, 24.93279036, 27.10831817, 26.88697449, 25.80286533,
       27.3116128 , 25.4060967 , 27.55552521, 28.30159717, 25.17681247,
       28.26655258, 27.82207145, 28.22772349, 30.45361189, 30.23373524,
       28.89402766, 30.98637061, 31.13245499, 29.01262802, 29.44403878,
       29.12494048, 31.06098902, 27.02074993, 30.87370365, 32.092818  ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a0200bb1-5afd-402c-8284-a6c1f4971599' class='xr-section-summary-in' type='checkbox'  ><label for='section-a0200bb1-5afd-402c-8284-a6c1f4971599' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>t</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-0f6569cc-0300-40f9-81f5-07b531e578bc' class='xr-index-data-in' type='checkbox'/><label for='index-0f6569cc-0300-40f9-81f5-07b531e578bc' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([                0.0, 0.10101010101010101, 0.20202020202020202,
       0.30303030303030304, 0.40404040404040403,  0.5050505050505051,
        0.6060606060606061,  0.7070707070707071,  0.8080808080808081,
        0.9090909090909091,  1.0101010101010102,  1.1111111111111112,
        1.2121212121212122,  1.3131313131313131,  1.4141414141414141,
        1.5151515151515151,  1.6161616161616161,  1.7171717171717171,
        1.8181818181818181,  1.9191919191919191,  2.0202020202020203,
         2.121212121212121,  2.2222222222222223,   2.323232323232323,
        2.4242424242424243,   2.525252525252525,  2.6262626262626263,
         2.727272727272727,  2.8282828282828283,   2.929292929292929,
        3.0303030303030303,   3.131313131313131,  3.2323232323232323,
        3.3333333333333335,  3.4343434343434343,  3.5353535353535355,
        3.6363636363636362,  3.7373737373737375,  3.8383838383838382,
        3.9393939393939394,   4.040404040404041,   4.141414141414141,
         4.242424242424242,   4.343434343434343,   4.444444444444445,
         4.545454545454545,   4.646464646464646,   4.747474747474747,
         4.848484848484849,    4.94949494949495,    5.05050505050505,
         5.151515151515151,   5.252525252525253,   5.353535353535354,
         5.454545454545454,   5.555555555555555,   5.656565656565657,
         5.757575757575758,   5.858585858585858,   5.959595959595959,
        6.0606060606060606,   6.161616161616162,   6.262626262626262,
         6.363636363636363,  6.4646464646464645,   6.565656565656566,
         6.666666666666667,   6.767676767676767,  6.8686868686868685,
          6.96969696969697,   7.070707070707071,   7.171717171717171,
        7.2727272727272725,   7.373737373737374,   7.474747474747475,
         7.575757575757575,  7.6767676767676765,   7.777777777777778,
         7.878787878787879,   7.979797979797979,   8.080808080808081,
         8.181818181818182,   8.282828282828282,   8.383838383838384,
         8.484848484848484,   8.585858585858587,   8.686868686868687,
         8.787878787878787,    8.88888888888889,    8.98989898989899,
          9.09090909090909,   9.191919191919192,   9.292929292929292,
         9.393939393939394,   9.494949494949495,   9.595959595959595,
         9.696969696969697,   9.797979797979798,     9.8989898989899,
                      10.0],
      dtype=&#x27;float64&#x27;, name=&#x27;t&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-65faa5e2-c491-48df-8cb4-0b387f00d0f0' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-65faa5e2-c491-48df-8cb4-0b387f00d0f0' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




    
![png](superquickstart_files/superquickstart_7_1.png)
    


## Initialize a simulation ✨

In pymob, a **simulation object** is initialized by creating an instance of the {class}`~pymob.simulation.SimulationBase` class from the simulation module.  
We will choose a linear regression model, as it provides a good approximation of the data: $ y = a + b*x $

```{admonition} x-dimension
:class: note
The x_dimension of our simulation can have any name, for example t as often used for time series data.
You can specify it via `sim.config.simulation.x_dimension`.
```


```python
# Initialize the Simulation object
sim = SimulationBase()

# configurate the case study
sim.config.case_study.name = "superquickstart"
sim.config.case_study.scenario = "linreg"

# Define the linear regression model
def linreg(x, a, b):
    return a + b * x

# Add the model to the simulation
sim.model = linreg

# Adding our dataset to the simulation
sim.observations = data_obs

# Defining a solver
sim.solver = solve_analytic_1d

# Take a look at the layut of the data
sim.config.data_structure
```

    MinMaxScaler(variable=y, min=-0.21899969389420804, max=32.09281799761304)
    

    C:\Pymob\pymob\pymob\simulation.py:307: UserWarning: `sim.config.data_structure.y = Datavariable(dimensions=['t'] min=-0.21899969389420804 max=32.09281799761304 observed=True dimensions_evaluator=None)` has been assumed from `sim.observations`. If the order of the dimensions should be different, specify `sim.config.data_structure.y = DataVariable(dimensions=[...], ...)` manually.
      warnings.warn(
    




    Datastructure(y=DataVariable(dimensions=['t'], min=-0.21899969389420804, max=32.09281799761304, observed=True, dimensions_evaluator=None))



```{admonition} Scalers
:class: note
We notice a mysterious Scaler message. This tells us that our data variable has been identified and a scaler was constructed, which transforms the variable between [0, 1].   
This has no effect at the moment, but it can be used later. Scaling can be powerful to help parameter estimation in more complex models.
```


## Parameterizing and running the model 🏃

Next, we define the **model parameters** $a$ and $b$.  
Parameter $a$ is set as fixed (`free = False`), meaning its value is known and will not be estimated during optimization.  
Parameter $b$ is marked as free (`free = True`), allowing it to be optimized to fit the data. As an initial guess, we assume $b = 3$.   


```python
# Parameterizing the model
sim.config.model_parameters.a = Param(value=1.0, free=False)
sim.config.model_parameters.b = Param(value=3.0, free=True)
# this makes sure the model parameters are available to the model.
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

sim.model_parameters["parameters"] 
```




    {'a': 1.0, 'b': 3.0}



Our model is now prepared with a defined parameter set.  
To initialize the **Evaluator**, we call {meth}`~pymob.simulation.SimulationBase.dispatch_constructor()`.   
This step is essential and must be executed every time changes are made to the model. 

The returned dataset (`evaluator.results`) has the exact same shape as the observation data.


```python
# put everything in place for running the simulation
sim.dispatch_constructor()

# run
evaluator = sim.dispatch(theta={"b":3})
evaluator()
evaluator.results
```

    C:\Pymob\pymob\pymob\simulation.py:567: UserWarning: The number of ODE states was not specified in the config file [simulation] > 'n_ode_states = <n>'. Extracted the return arguments ['a+b*x'] from the source code. Setting 'n_ode_states=1.
      warnings.warn(
    




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base rgba(0, 0, 0, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, white)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
  );
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base, rgba(255, 255, 255, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, #111111)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
  );
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
  border: 2px solid transparent !important;
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0) !important;
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  border-color: var(--xr-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
  border-color: var(--xr-background-color-row-even);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  border-top: 2px dotted var(--xr-background-color);
  padding-bottom: 20px !important;
  padding-top: 10px !important;
}

.xr-var-attrs-in + label,
.xr-var-data-in + label,
.xr-index-data-in + label {
  padding: 0 1px;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-data > pre,
.xr-index-data > pre,
.xr-var-data > table > tbody > tr {
  background-color: transparent !important;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}

.xr-var-attrs-in:checked + label > .xr-icon-file-text2,
.xr-var-data-in:checked + label > .xr-icon-database,
.xr-index-data-in:checked + label > .xr-icon-database {
  color: var(--xr-font-color0);
  filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
  stroke-width: 0.8px;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 2kB
Dimensions:  (t: 100)
Coordinates:
  * t        (t) float64 800B 0.0 0.101 0.202 0.303 ... 9.697 9.798 9.899 10.0
Data variables:
    y        (t) float64 800B 1.0 1.303 1.606 1.909 ... 30.09 30.39 30.7 31.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-843c93d5-f2e9-4819-9f89-2ea8de587d95' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-843c93d5-f2e9-4819-9f89-2ea8de587d95' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>t</span>: 100</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-8dae8c55-0d33-4a1c-ac84-9e679dd2d172' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8dae8c55-0d33-4a1c-ac84-9e679dd2d172' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>t</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.101 0.202 ... 9.899 10.0</div><input id='attrs-e0a4798e-87be-49d8-99e3-be36163fb43f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e0a4798e-87be-49d8-99e3-be36163fb43f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-846ef747-2ea0-4f0b-8375-fbd69975f778' class='xr-var-data-in' type='checkbox'><label for='data-846ef747-2ea0-4f0b-8375-fbd69975f778' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.      ,  0.10101 ,  0.20202 ,  0.30303 ,  0.40404 ,  0.505051,
        0.606061,  0.707071,  0.808081,  0.909091,  1.010101,  1.111111,
        1.212121,  1.313131,  1.414141,  1.515152,  1.616162,  1.717172,
        1.818182,  1.919192,  2.020202,  2.121212,  2.222222,  2.323232,
        2.424242,  2.525253,  2.626263,  2.727273,  2.828283,  2.929293,
        3.030303,  3.131313,  3.232323,  3.333333,  3.434343,  3.535354,
        3.636364,  3.737374,  3.838384,  3.939394,  4.040404,  4.141414,
        4.242424,  4.343434,  4.444444,  4.545455,  4.646465,  4.747475,
        4.848485,  4.949495,  5.050505,  5.151515,  5.252525,  5.353535,
        5.454545,  5.555556,  5.656566,  5.757576,  5.858586,  5.959596,
        6.060606,  6.161616,  6.262626,  6.363636,  6.464646,  6.565657,
        6.666667,  6.767677,  6.868687,  6.969697,  7.070707,  7.171717,
        7.272727,  7.373737,  7.474747,  7.575758,  7.676768,  7.777778,
        7.878788,  7.979798,  8.080808,  8.181818,  8.282828,  8.383838,
        8.484848,  8.585859,  8.686869,  8.787879,  8.888889,  8.989899,
        9.090909,  9.191919,  9.292929,  9.393939,  9.494949,  9.59596 ,
        9.69697 ,  9.79798 ,  9.89899 , 10.      ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e40a3c57-c1c9-4619-b14d-8a6db46c8dfe' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e40a3c57-c1c9-4619-b14d-8a6db46c8dfe' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.0 1.303 1.606 ... 30.39 30.7 31.0</div><input id='attrs-454c1729-5e7a-457c-9212-b2f6447d7024' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-454c1729-5e7a-457c-9212-b2f6447d7024' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-54742c46-e2f7-4d6c-b8ca-0ce1fdb9f1c4' class='xr-var-data-in' type='checkbox'><label for='data-54742c46-e2f7-4d6c-b8ca-0ce1fdb9f1c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 1.        ,  1.3030303 ,  1.60606061,  1.90909091,  2.21212121,
        2.51515152,  2.81818182,  3.12121212,  3.42424242,  3.72727273,
        4.03030303,  4.33333333,  4.63636364,  4.93939394,  5.24242424,
        5.54545455,  5.84848485,  6.15151515,  6.45454545,  6.75757576,
        7.06060606,  7.36363636,  7.66666667,  7.96969697,  8.27272727,
        8.57575758,  8.87878788,  9.18181818,  9.48484848,  9.78787879,
       10.09090909, 10.39393939, 10.6969697 , 11.        , 11.3030303 ,
       11.60606061, 11.90909091, 12.21212121, 12.51515152, 12.81818182,
       13.12121212, 13.42424242, 13.72727273, 14.03030303, 14.33333333,
       14.63636364, 14.93939394, 15.24242424, 15.54545455, 15.84848485,
       16.15151515, 16.45454545, 16.75757576, 17.06060606, 17.36363636,
       17.66666667, 17.96969697, 18.27272727, 18.57575758, 18.87878788,
       19.18181818, 19.48484848, 19.78787879, 20.09090909, 20.39393939,
       20.6969697 , 21.        , 21.3030303 , 21.60606061, 21.90909091,
       22.21212121, 22.51515152, 22.81818182, 23.12121212, 23.42424242,
       23.72727273, 24.03030303, 24.33333333, 24.63636364, 24.93939394,
       25.24242424, 25.54545455, 25.84848485, 26.15151515, 26.45454545,
       26.75757576, 27.06060606, 27.36363636, 27.66666667, 27.96969697,
       28.27272727, 28.57575758, 28.87878788, 29.18181818, 29.48484848,
       29.78787879, 30.09090909, 30.39393939, 30.6969697 , 31.        ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-963b279c-24d3-4340-a7db-d217683908c5' class='xr-section-summary-in' type='checkbox'  ><label for='section-963b279c-24d3-4340-a7db-d217683908c5' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>t</div></div><div class='xr-index-preview'>PandasIndex</div><input type='checkbox' disabled/><label></label><input id='index-4a0e9593-b218-4999-a36a-0b6a7108a6ca' class='xr-index-data-in' type='checkbox'/><label for='index-4a0e9593-b218-4999-a36a-0b6a7108a6ca' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([                0.0, 0.10101010101010101, 0.20202020202020202,
       0.30303030303030304, 0.40404040404040403,  0.5050505050505051,
        0.6060606060606061,  0.7070707070707071,  0.8080808080808081,
        0.9090909090909091,  1.0101010101010102,  1.1111111111111112,
        1.2121212121212122,  1.3131313131313131,  1.4141414141414141,
        1.5151515151515151,  1.6161616161616161,  1.7171717171717171,
        1.8181818181818181,  1.9191919191919191,  2.0202020202020203,
         2.121212121212121,  2.2222222222222223,   2.323232323232323,
        2.4242424242424243,   2.525252525252525,  2.6262626262626263,
         2.727272727272727,  2.8282828282828283,   2.929292929292929,
        3.0303030303030303,   3.131313131313131,  3.2323232323232323,
        3.3333333333333335,  3.4343434343434343,  3.5353535353535355,
        3.6363636363636362,  3.7373737373737375,  3.8383838383838382,
        3.9393939393939394,   4.040404040404041,   4.141414141414141,
         4.242424242424242,   4.343434343434343,   4.444444444444445,
         4.545454545454545,   4.646464646464646,   4.747474747474747,
         4.848484848484849,    4.94949494949495,    5.05050505050505,
         5.151515151515151,   5.252525252525253,   5.353535353535354,
         5.454545454545454,   5.555555555555555,   5.656565656565657,
         5.757575757575758,   5.858585858585858,   5.959595959595959,
        6.0606060606060606,   6.161616161616162,   6.262626262626262,
         6.363636363636363,  6.4646464646464645,   6.565656565656566,
         6.666666666666667,   6.767676767676767,  6.8686868686868685,
          6.96969696969697,   7.070707070707071,   7.171717171717171,
        7.2727272727272725,   7.373737373737374,   7.474747474747475,
         7.575757575757575,  7.6767676767676765,   7.777777777777778,
         7.878787878787879,   7.979797979797979,   8.080808080808081,
         8.181818181818182,   8.282828282828282,   8.383838383838384,
         8.484848484848484,   8.585858585858587,   8.686868686868687,
         8.787878787878787,    8.88888888888889,    8.98989898989899,
          9.09090909090909,   9.191919191919192,   9.292929292929292,
         9.393939393939394,   9.494949494949495,   9.595959595959595,
         9.696969696969697,   9.797979797979798,     9.8989898989899,
                      10.0],
      dtype=&#x27;float64&#x27;, name=&#x27;t&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4a6abe8f-ee48-42ef-a2ab-7f3e31f0ecdb' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-4a6abe8f-ee48-42ef-a2ab-7f3e31f0ecdb' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



```{admonition} What does the dispatch constructor do?
:class: hint
Behind the scenes, the dispatch constructor assembles a lightweight Evaluator object from the Simulation object, that takes the least necessary amount of information, runs it through some dimension checks, and also connects it to the specified solver and initializes it.
```

Let's take a look at the **results**.  

You can vary the parameter $b$ in the previous step to investigate its influence on the model fit.  
In the [Introduction](https://pymob.readthedocs.io/en/stable/user_guide/introduction.html), you can try out the *manual parameter estimation*, which is a feature provided by Pymob.  


```python
fig, ax = plt.subplots(figsize=(5, 4))
data_res = evaluator.results
ax.plot(data_obs.t, data_obs.y, ls="", marker="o", color="tab:blue", alpha=.5, label ="observation data")
ax.plot(data_res.t, data_res.y, color="black", label ="result")
ax.legend()
```




    <matplotlib.legend.Legend at 0x1df6d279990>




    
![png](superquickstart_files/superquickstart_18_1.png)
    


## Estimating parameters and uncertainty with MCMC 🤔
Of course this example is very simple. In fact, we could optimize the parameters perfectly by hand.   
But just for fun, let's use *Markov Chain Monte Carlo (MCMC)* to estimate the parameters, their uncertainty and the uncertainty in the data.   
We’ll run the parameter estimation with our **{attr}`~pymob.simulation.inferer`**, using the NumPyro backend with a NUTS kernel. This completes the job in a few seconds.

We are almost ready to infer the model parameters. To also estimate the uncertainty of the parameters, we add another parameter representing the error and assume that it follows a lognormal distribution.   
Additionally, we specify an error model for the data distribution. This will be: $$y_{obs} \sim Normal (y, \sigma_y)$$  

Since $\sigma_y$ is not a fixed parameter, it doesn't need to be passed to the simulation class.


```python
sim.config.model_parameters.sigma_y = Param(free=True , prior="lognorm(scale=1,s=1)", min=0, max=1)
sim.config.model_parameters.b.prior = "lognorm(scale=1,s=1)"

sim.config.error_model.y = "normal(loc=y,scale=sigma_y)"


sim.set_inferer("numpyro")
sim.inferer.config.inference_numpyro.kernel = "nuts"
sim.inferer.run()

sim.inferer.idata.posterior

# Plot the results
sim.config.simulation.x_dimension = "t"
sim.posterior_predictive_checks(pred_hdi_style={"alpha": 0.1})
```

    Jax 64 bit mode: False
    Absolute tolerance: 1e-07
    

    Trace Shapes:      
     Param Sites:      
    Sample Sites:      
           b dist     |
            value     |
     sigma_y dist     |
            value     |
       y_obs dist 100 |
            value 100 |
    

      0%|                                                                                                                                                                                                               | 0/3000 [00:00<?, ?it/s]

    warmup:   0%|                                                                                                                                                     | 1/3000 [00:01<51:56,  1.04s/it, 1 steps of size 1.87e+00. acc. prob=0.00]

    warmup:  10%|███████████████▎                                                                                                                                  | 315/3000 [00:01<00:07, 375.41it/s, 3 steps of size 5.98e-01. acc. prob=0.78]

    warmup:  20%|█████████████████████████████▉                                                                                                                    | 614/3000 [00:01<00:03, 755.62it/s, 7 steps of size 8.67e-01. acc. prob=0.79]

    warmup:  31%|████████████████████████████████████████████▌                                                                                                    | 922/3000 [00:01<00:01, 1153.67it/s, 1 steps of size 8.99e-01. acc. prob=0.79]

    sample:  41%|███████████████████████████████████████████████████████████▋                                                                                    | 1244/3000 [00:01<00:01, 1538.23it/s, 3 steps of size 8.50e-01. acc. prob=0.92]

    sample:  52%|██████████████████████████████████████████████████████████████████████████▍                                                                     | 1552/3000 [00:01<00:00, 1867.98it/s, 3 steps of size 8.50e-01. acc. prob=0.92]

    sample:  61%|████████████████████████████████████████████████████████████████████████████████████████▏                                                       | 1836/3000 [00:01<00:00, 2094.01it/s, 7 steps of size 8.50e-01. acc. prob=0.91]

    sample:  74%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                     | 2210/3000 [00:01<00:00, 2416.20it/s, 11 steps of size 8.50e-01. acc. prob=0.91]

    sample:  86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                   | 2595/3000 [00:01<00:00, 2743.24it/s, 3 steps of size 8.50e-01. acc. prob=0.91]

    sample: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉| 2999/3000 [00:01<00:00, 3005.45it/s, 3 steps of size 8.50e-01. acc. prob=0.91]

    sample: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:01<00:00, 1505.68it/s, 1 steps of size 8.50e-01. acc. prob=0.91]

    
    

    
                    mean       std    median      5.0%     95.0%     n_eff     r_hat
             b      3.07      0.03      3.07      3.02      3.11   2085.24      1.00
       sigma_y      1.57      0.12      1.57      1.38      1.76   1661.94      1.00
    
    Number of divergences: 0
    


    
![png](superquickstart_files/superquickstart_20_16.png)
    


```{admonition} numpyro distributions
:class: warning
Currently only few distributions are implemented in the numpyro backend. This API will soon change, so that basically any distribution can be used to specifcy parameters. 
```

We can **inspect our estimates** and see that the model provides a good fit for the parameters.  
Note that we only get an estimate for $b$. Previously, we set the parameter $a$ with the flag `free = False`.   
This effectively excludes it from the estimation and uses its default value, which was set to the true value `a = 0`.


```{admonition} Customize the posterior predictive checks
:class: hint
You can explore the API of {class}`pymob.sim.plot.SimulationPlot` to find out how you can work on the default predictions. Of course you can always make your own plot, by accessing {attr}`pymob.simulation.inferer.idata` and {attr}`pymob.simulation.observations`
```

## Report the results 🗒️

Pymob provides the option to generate an automated report of the parameter distribution for a simulation.  
The report can be configured by modifying the options in {meth}`~pymob.simulation.SimulationBase.config.report`.


```python
# report the results
sim.report()
```

![posterior_trace.png](superquickstart_files/posterior_trace.png)

![posterior_pairs.png](superquickstart_files/posterior_pairs.png)


## Exporting the simulation and running it via the case study API 📤

After constructing the simulation, all settings - custom and default - can be exported to a comprehensive configuration file.   
The simulation will be saved to the default path (`CASE_STUDY/scenarios/SCENARIO/settings.cfg`) or to a custom path, specified with the file path keyword `fp`.   
Setting `force=True` will overwrite any existing config file, which is a reasonable choice in most cases.
From this point on, the simulation is (almost) ready to be executed from the command-line. 


```python
import os
sim.config.create_directory("scenario", force=True)
sim.config.create_directory("results", force=True)

# usually we expect to have a data directory in the case
os.makedirs(sim.data_path, exist_ok=True)
sim.save_observations(force=True)
sim.config.save(force=True)
```

    Scenario directory exists at 'C:\Users\mgrho\pymob\docs\source\user_guide\case_studies\superquickstart\scenarios\linreg'.
    Results directory exists at 'C:\Users\mgrho\pymob\docs\source\user_guide\case_studies\superquickstart\results\linreg'.
    

### Commandline API

The command-line API runs a series of commands that load the case study, execute the {meth}`~pymob.simulation.SimulationBase.initialize` method and perform some more initialization tasks before running the required job.

+ `pymob-infer` runs an inference job, for example:  

  `pymob-infer --case_study=quickstart --scenario=test --inference_backend=numpyro`.   
  While there are more command-line options, these two (--case_study and --scenario) are required.



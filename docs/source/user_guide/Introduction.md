# Pymob  Introduction
## Overview
**Pymob** is a Python-based platform for parameter estimation across a wide range of models. It abstracts repetitive tasks in the modeling process so that you can focus on building models, asking questions to the real world and learn from observations. <br>
The idea of pymob originated from the frustration with fitting complex models to complicated datasets (missing observations, non-uniform data structure, non-linear models, ODE models). In such scenarios a lot of time is spent matching observations with model results. <br>
One of Pymob’s key strengths is its streamlined model definition workflow. This not only simplifies the process of building models but also lets you apply a host of advanced optimization and inference algorithms, giving you the flexibility to iterate and discover solutions more effectively. <br>

### What's the focus of this introduction?
This introduction will give you an overview of the pymob package and an easy example on how to use it. After, you can explore more advanced tutorials and deepen your pymob kowledge. <br> 
First the general structure of the pymob package will be explained. You will get to know the function of the components. Subsequentenly you will get instructions to use pymob for your first parameter estimation with a simple example. 

### How pymob is structured:
Here  you can see the structure of the structure of pymob package: <br>
![Structure of the pymob package](./figures/pymob_overview.png) <br>
The Pymob package consists of several elements: 


1) __Simulation__ <br>
First, we need to initialize a Simulation object by calling the {class}`pymob.simulation.SimulationBase` class from the simulation module.   
Optionally, we can configure the simulation object with {attr}`pymob.simulation.SimulationBase.config.case_study.name` = "linear-regression", {attr}`pymob.simulation.SimulationBase.config.case_study.scenario` = "test" and many more options. 

2) __Model__ <br>
The model is a python function you define. With the model you try to describe the data you observed. A classical model is, for example, the Lotka-Volterra model to describe the interactions of predators and prey. In the tutorial today, the model will be a simple linear function. <br>
The model will be added to the simualtion by using {class}`pymob.simulation.SimulationBase.model`

3) __Observations__ <br>
The obseravtions are the data points, to which we want to fit our model. The observation data needs to be an `xarray.Dataset` ([learn more here](https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html)).  
We assign it to our Simulation object by  {attr}`pymob.simulation.SimulationBase.observations`.  
{attr}`pymob.simulation.SimulationBase.config.data_structure` will give us some information about the layout of our data.

4) __Solver__ <br>
A solver is required for many models e.g. models that contain differential equations. Solvers in pymob are callables that need to return a dictionary of results mapped to the data variables. <br>
The solver is assigned to the Simulation object by {class}`pymob.simulation.SimulationBase.solver`. <br>
These solvers are currently implemented in pymob: 
    - analytic module
        - solve_analytic_1d
    - base module 
        - curve_jumps
        - jump_interpolation
        - mappar
        - radius_interpolation
        - rect_interpolation
        - smoothed_interpolation
    - diffrax module
        - JaxSolver
    - scipy module
        - solve_ivp_1d

The documentation can be found [here](https://pymob.readthedocs.io/en/stable/api/pymob.solvers.html) 

5) __Inferer__ <br>
    The inferer serves as the parameter estimator. Pymob provides various backends. You can find detailed information [here](https://pymob.readthedocs.io/en/stable/user_guide/framework_overview.html). <br>
    Currently, supported inference backends are:
    * interactive (interactive backend in jupyter notebookswith parameter sliders)
    * numpyro (bayesian inference and stochastic variational inference)
    * pyabc (approximate bayesian inference)
    * pymoo (experimental multi-objective optimization)

6) __Evaluator__ <br>
The Evaluator is an instance to manage model evaluations. It sets up tasks, coordinates parallel runs of the simulation and keeps track of the results from each simulation or parameter inference process.

7) __Config__ <br>
Pymob uses `pydantic` models to validate configuration files, with the configuration organized into separate sections. You can modify these configurations either by editing the files before initializing a simulation from a config file, or directly within the script. During parameter estimation setup, all configuration settings are stored in a config object, which can later be exported as a `.cfg` file.








### Let's get started 🎉
You will need several packages during this introduction:


```python
# imports from pymob
from pymob.simulation import SimulationBase
from pymob.sim.solvetools import solve_analytic_1d
from pymob.sim.config import Param

# other imports
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import os
from numpy import random
```

In the following tutorial, you’ll notice some import statements included as comments. These are provided to indicate which package is required for each step.

## Generate artificial data

In the real world, you will have measured a dataset. For demonstration, we generate some artifical data. Later we will fit the model to our artifical data. <br>
$y_{obs}$ represents the observation data over the time $t$ [0, 10].  


```python
# Parameter for the artificial data generation
rng = np.random.default_rng(seed=1)  # for reproducibility
slope = rng.uniform(1,4)
intercept = 1.0
num_points = 100
noise_level = 1.7

# generating x-values
x = np.linspace(0, 10, num_points)

# generating y-values with noise
noise = rng.normal(0, noise_level, num_points)
y_obs = slope * x + intercept + noise

data = np.array(y_obs)

# visualising our data
plt.scatter(x, y_obs, label='Datapoints')
plt.xlabel('t [-]')
plt.ylabel('y_obs [-]')
plt.title('Artificial Data')
plt.legend()
plt.show()
```


    
![png](Introduction_files/Introduction_4_0.png)
    


Above you can see you're generated artificial data. At the moment it's stored in a normal array as you can see below: 


```python
# our artificial data is now in the variable data
print(data)
```

    [ 2.39675084  1.81785059 -0.70315217  3.30742766  2.78326703  1.36771732
      3.52454616  3.41252601  3.54888575  3.35328588  4.49048771  2.56521125
      3.79634384  3.50979549  5.60354444  4.90914103  4.60054453  4.02458419
      5.17270933  5.8798854   5.65362632  8.57816731  8.34579772  2.28149774
      3.93525899  7.10557652  6.94107294  8.2780973   8.54045905 12.02744521
      6.79279159  8.29740594 12.66815375 10.55094467 10.83486488  9.08995387
      7.41814448 10.7606699  10.91741134  8.90169647 10.0828172  11.37793583
     10.15043989 11.84556627 12.43105392 12.58533694 11.92025208 14.04642718
     14.80814685 14.09471271 12.41438677 15.3052946  13.46514525 16.06827389
     13.0077698  16.64051021 15.30791566 13.47525798 15.32060955 16.20232009
     16.83019906 14.95284153 14.99613473 17.47407018 16.59740969 18.04735114
     19.19428235 15.3562682  18.84777408 20.75332169 18.42173378 17.80525218
     20.71855905 20.12671118 21.47496089 19.62120052 17.94508373 20.53326405
     20.21848206 22.55054798 21.81778089 18.97226891 19.96904293 23.75936909
     23.66863583 21.68072914 23.02346747 24.03883303 24.33375292 25.28318484
     24.48570624 24.14458006 24.12185409 26.61276612 21.24765866 25.09450444
     25.64242623 23.41934038 26.66432432 25.24747102]


The pymob package operates with `xarray.Dataset`. We avoid most of the mess by using `xarray` as a common input/output format. So we have to transform our data into a `xarray.Dataset`.


```python
obs_data = xr.DataArray(data, dims = ("t"), coords={"t": x}).to_dataset(name="data") 
```

Note: If you want to rename your data-dimension you have to change every {class}`sim.config.data_structure.data` to the new name!

It can be helpful to look at the data befor going forward, especially if you never worked with *xarray Datasets*. At the section 'Data variables' you'll find the data you just generated. 


```python
obs_data
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
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
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
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
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
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
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
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
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
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
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
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
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
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (t: 100)
Coordinates:
  * t        (t) float64 0.0 0.101 0.202 0.303 0.404 ... 9.697 9.798 9.899 10.0
Data variables:
    data     (t) float64 2.397 1.818 -0.7032 3.307 ... 25.64 23.42 26.66 25.25</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-1fc3766b-2e18-4856-bb27-5c98bd62d264' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-1fc3766b-2e18-4856-bb27-5c98bd62d264' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>t</span>: 100</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-f93fa6f3-7d1b-4e48-9a11-f8ebdae878c0' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f93fa6f3-7d1b-4e48-9a11-f8ebdae878c0' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>t</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.101 0.202 ... 9.899 10.0</div><input id='attrs-6c391e33-ed2d-4111-b15c-5ecbd792ff32' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6c391e33-ed2d-4111-b15c-5ecbd792ff32' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4d5ce10d-712f-465d-96b5-9b71637159fa' class='xr-var-data-in' type='checkbox'><label for='data-4d5ce10d-712f-465d-96b5-9b71637159fa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.      ,  0.10101 ,  0.20202 ,  0.30303 ,  0.40404 ,  0.505051,
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
        9.69697 ,  9.79798 ,  9.89899 , 10.      ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-fbb846bb-b7dc-4576-96a0-fabe376591ea' class='xr-section-summary-in' type='checkbox'  checked><label for='section-fbb846bb-b7dc-4576-96a0-fabe376591ea' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>data</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.397 1.818 -0.7032 ... 26.66 25.25</div><input id='attrs-da4de1cd-2cfe-44e7-b9a4-d75c50913050' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-da4de1cd-2cfe-44e7-b9a4-d75c50913050' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-062e81aa-3ab8-4a21-a7b9-108eb2a59b20' class='xr-var-data-in' type='checkbox'><label for='data-062e81aa-3ab8-4a21-a7b9-108eb2a59b20' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 2.39675084,  1.81785059, -0.70315217,  3.30742766,  2.78326703,
        1.36771732,  3.52454616,  3.41252601,  3.54888575,  3.35328588,
        4.49048771,  2.56521125,  3.79634384,  3.50979549,  5.60354444,
        4.90914103,  4.60054453,  4.02458419,  5.17270933,  5.8798854 ,
        5.65362632,  8.57816731,  8.34579772,  2.28149774,  3.93525899,
        7.10557652,  6.94107294,  8.2780973 ,  8.54045905, 12.02744521,
        6.79279159,  8.29740594, 12.66815375, 10.55094467, 10.83486488,
        9.08995387,  7.41814448, 10.7606699 , 10.91741134,  8.90169647,
       10.0828172 , 11.37793583, 10.15043989, 11.84556627, 12.43105392,
       12.58533694, 11.92025208, 14.04642718, 14.80814685, 14.09471271,
       12.41438677, 15.3052946 , 13.46514525, 16.06827389, 13.0077698 ,
       16.64051021, 15.30791566, 13.47525798, 15.32060955, 16.20232009,
       16.83019906, 14.95284153, 14.99613473, 17.47407018, 16.59740969,
       18.04735114, 19.19428235, 15.3562682 , 18.84777408, 20.75332169,
       18.42173378, 17.80525218, 20.71855905, 20.12671118, 21.47496089,
       19.62120052, 17.94508373, 20.53326405, 20.21848206, 22.55054798,
       21.81778089, 18.97226891, 19.96904293, 23.75936909, 23.66863583,
       21.68072914, 23.02346747, 24.03883303, 24.33375292, 25.28318484,
       24.48570624, 24.14458006, 24.12185409, 26.61276612, 21.24765866,
       25.09450444, 25.64242623, 23.41934038, 26.66432432, 25.24747102])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4a67b2d0-16a8-4865-8dcc-8011464cd1c9' class='xr-section-summary-in' type='checkbox'  ><label for='section-4a67b2d0-16a8-4865-8dcc-8011464cd1c9' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>t</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-d694207c-cfb6-4d7a-86e3-cd8c91d40c7d' class='xr-index-data-in' type='checkbox'/><label for='index-d694207c-cfb6-4d7a-86e3-cd8c91d40c7d' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([                0.0, 0.10101010101010101, 0.20202020202020202,
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
      dtype=&#x27;float64&#x27;, name=&#x27;t&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-02d16bd6-226f-4835-beb9-6e2b0eb5312a' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-02d16bd6-226f-4835-beb9-6e2b0eb5312a' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



## Initialize a simulation
First, we initialize an object of the class simulation. This is the center of the whole package and will manage all processes from now on. <br>
In pymob a Simulation object is initialized by calling the {class}`pymob.simulation.SimulationBase` class from the simulation module.


```python
#from pymob.simulation import SimulationBase

sim = SimulationBase()
```

```{admonition} Configuring the simulation
:class: note
Optionally, we can configure the simulation at this stage with 
`sim.config.case_study.name = "linear-regression"`, `sim.config.case_study.scenario = "test"`, and many more options. 
```
Case studies are a principled approach to the modelling process. In essence, they are a simple template that contains building blocks for model and names and stores them in an intuitive and reproducible way. [Here](https://pymob.readthedocs.io/en/stable/user_guide/case_studies.html#configuration) you'll find some additional information on case studies. <br>

At the moment, it is sufficient to only create a simulation object without making any further configurations.

## Define a model 

Now the model needs to be defined. In Pymob, every model is represented as a Python function. Here, you’ll specify the model whose parameters will be estimated.

In this tutorial, we’ll use linear regression as our example, since it’s the simplest form of modeling.


```python
# definition of the model: 
def linreg(t, a, b):
    return a + t * b
```

So we assume that this model describes our data well. So we add it to the simulation by


```python
sim.model = linreg
```


## Defining a solver

As described above: A solver is required for many models. So we define a solver by {class}`pymob.simulation.SimulationBase.solver`. <br>
In our case the model gives the exact solution of the model. Therefore, we choose `solve_analytic_1d`. An overwiev of the solvers currently implemented in pymob can be found at the beginning of this tutorial [here](#how-pymob-is-structured).


```python
# from pymob.sim.solvetools import solve_analytic_1d
sim.solver = solve_analytic_1d
```

## The pymob magic

So far we have not done anything special. Pymob exists, because wrangling dimensions of input and output data, nested data-structures, missing data is painful. <br>

Now we add our data, which is already transformed into a *xarray Dataset*, by using {attr}`pymob.simulation.SimulationBase.observations`.


```python
# import xarray as xr

sim.observations = obs_data
```

    MinMaxScaler(variable=data, min=-0.7031521676464498, max=26.6643243203019)


    /export/home/fschunck/miniconda3/envs/pymob/lib/python3.11/site-packages/pymob/simulation.py:361: UserWarning: `sim.config.data_structure.data = Datavariable(dimensions=['t'] min=-0.7031521676464498 max=26.6643243203019 observed=True dimensions_evaluator=None)` has been assumed from `sim.observations`. If the order of the dimensions should be different, specify `sim.config.data_structure.data = DataVariable(dimensions=[...], ...)` manually.
      warnings.warn(


This worked 🎉 {attr}`pymob.simulation.SimulationBase.config.data_structure` will now give us some information about the layout of our data, which will handle the data transformations in the background.


```python
sim.config.data_structure
```




    Datastructure(data=DataVariable(dimensions=['t'], min=-0.7031521676464498, max=26.6643243203019, observed=True, dimensions_evaluator=None))



```{admonition} What happens when we assign a Dataset to the observations attribute?
:class: hint

Debug into the function and discover what happens!
```

We can give `pymob` additional information about the data structure of our observations and intermediate (unobserved) variables that are simulated. This can be done with {attr}`sim.config.data_structure.y` = `DataVariable(dimensions=["x"])`.
These information can be used to switch the dimensional order of the observations or provide data variables that have differing dimensions from the observations, if needed. But if the dataset is ordinary, simply setting {attr}`pymob.simulation.SimulationBase.observations` property with a `xr.Dataset` will be sufficient.

```{admonition} Scalers
:class: note
We also notice a mysterious Scaler message. This tells us that our data variable has been identified and a scaler was constructed, which transforms the variable between [0, 1]. This has no effect at the moment, but it can be used later. Scaling can be powerful to help parameter estimation in more complex models.
```

## Parameterizing a model

Parameters are specified via the `FloatParam` or `ArrayParam` class. Parameters can be marked free or fixed depending on whether they should be variable during an optimization procedure. <br>

In this tutorial we want to fit the parameter $b$ and assume that we know parameter $a$: <br>
* The parameter $a$ is set as fixed (`free = False`), meaning its value is known and will not be estimated during optimization.
* The parameter $b$ is marked as free (`free = True`), allowing it to be optimized to fit our data. As an initial guess, we assume $b = 3$.



```python
#from pymob.sim.config import Param
sim.config.model_parameters.a = Param(value=0, free=False)
sim.config.model_parameters.b = Param(value=3, free=True)

# this makes sure the model parameters are available to the model.
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
```

To make the parameters available to the simulation one has to use {attr}`sim.model_parameters["parameters"]` = {attr}`sim.config.model_parameters.value_dict`. This step is particularly important for all fixed parameters.

{attr}`pymob.simulation.SimulationBase.model_parameters` is a dictionary that stores the input data for the model. By default, it includes the keys `parameters`, `y0`, and `x_in`. For our analytic model, we only need the `parameters` key. In situations where initial values for variables are required, you can provide them using {attr}`pymob.simulation.SimulationBase.model_parameters["y0"]` = ... .

For example, when working with a Lotka-Volterra model, you would specify the initial conditions for the predator and prey populations with `y0`. For more details on such use cases, please refer to the advanced tutorial.

```{admonition} generating input for solvers
:class: note
A helpful function to generate `y0` or `x_in` from observations is `SimulationBase.parse_input`, combined with settings of `config.simulation.y0`
```


```python
sim.model_parameters['parameters']
```




    {'a': array(0), 'b': array(3)}



## Running the model 🏃

The model is prepared with a parameter set and ready to be executed. With {class}`pymob.simulation.SimulationBase.dispatch_constructor()`, everything is prepared for the run of the model. It initiaizes an `evaluator`, makes preliminary calculations and checks. 

ℹ️ What does the dispatch constructor do?: <br>
Behind the scenes, the dispatch constructor assembles a lightweight {class}`pymob.simulation.SimulationBase.evaluator` object from the Simulation object, that takes the least necessary amount of information, runs it through some dimension checks, and also connects it to the specified solver and initializes it. The purpose of the dispatch constructor is manyfold: <br>
By executing the entire overhead of a model evaluation and packing it into a new {class}`pymob.simulation.SimulationBase.evaluator` instance {meth}`pymob.simulation.SimulationBase.dispatch_constructor()` to make single model evaluations as fast as possible and allow parallel evaluations, because each evaluator created by {meth}`pymob.simulation.SimulationBase.dispatch()` is it's a fully independent model instance with a separate set of parameters that can be solved.
Evaluators store the raw output from a simulation and can generate an xarray object from it that corresponds to the data-structure of the observations with the {attr}`pymob.simulation.SimulationBase.evaluator.results` property. This automatically aligns simulations results with observations, for simple computation of loss functions.

For the parameter estimation it is not necessary to run the model, but it can be helpfull. By using {meth}`pymob.simulation.SimulationBase.dispatch()` all the parameters with the setting `free=True` get fixed. Therefore, we have to fix parameter $b$. 

*Try changing the value of $b$ and see what effect it has on the next steps?* <br>

**{meth}`pymob.simulation.SimulationBase.dispatch_constructor()` should be executed every time you change something in your simulation settings, even if you don't run the model.** <br>


```python
# put everything in place for running the simulation
sim.dispatch_constructor()

# run
evaluator = sim.dispatch(theta={"b":3}) # makes sure that the parameter b is set to 3
evaluator()
evaluator.results
```

    /export/home/fschunck/miniconda3/envs/pymob/lib/python3.11/site-packages/pymob/simulation.py:688: UserWarning: The number of ODE states was not specified in the config file [simulation] > 'n_ode_states = <n>'. Extracted the return arguments ['a+t*b'] from the source code. Setting 'n_ode_states=1.
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
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
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
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
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
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
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
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
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
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
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
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
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
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (t: 100)
Coordinates:
  * t        (t) float64 0.0 0.101 0.202 0.303 0.404 ... 9.697 9.798 9.899 10.0
Data variables:
    data     (t) float64 0.0 0.303 0.6061 0.9091 1.212 ... 29.09 29.39 29.7 30.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-0a0d1bc7-799e-44a4-b9a3-863160895571' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-0a0d1bc7-799e-44a4-b9a3-863160895571' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>t</span>: 100</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-31f5815f-368c-497f-972e-fdb187022fb1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-31f5815f-368c-497f-972e-fdb187022fb1' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>t</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.101 0.202 ... 9.899 10.0</div><input id='attrs-d6f34d70-239d-4ab7-9b54-e4994237bda7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d6f34d70-239d-4ab7-9b54-e4994237bda7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b8beeaa6-d5a0-4226-a3f5-b6be9f7bc560' class='xr-var-data-in' type='checkbox'><label for='data-b8beeaa6-d5a0-4226-a3f5-b6be9f7bc560' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.      ,  0.10101 ,  0.20202 ,  0.30303 ,  0.40404 ,  0.505051,
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
        9.69697 ,  9.79798 ,  9.89899 , 10.      ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c88999a8-2214-4a9a-80d4-096cefb4ff7f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c88999a8-2214-4a9a-80d4-096cefb4ff7f' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>data</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.303 0.6061 ... 29.7 30.0</div><input id='attrs-eb1322bf-94a2-4843-b5d5-b9cc411e7669' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eb1322bf-94a2-4843-b5d5-b9cc411e7669' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-96f5832e-7c02-498c-86af-bf6b16b4c0af' class='xr-var-data-in' type='checkbox'><label for='data-96f5832e-7c02-498c-86af-bf6b16b4c0af' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.        ,  0.3030303 ,  0.60606061,  0.90909091,  1.21212121,
        1.51515152,  1.81818182,  2.12121212,  2.42424242,  2.72727273,
        3.03030303,  3.33333333,  3.63636364,  3.93939394,  4.24242424,
        4.54545455,  4.84848485,  5.15151515,  5.45454545,  5.75757576,
        6.06060606,  6.36363636,  6.66666667,  6.96969697,  7.27272727,
        7.57575758,  7.87878788,  8.18181818,  8.48484848,  8.78787879,
        9.09090909,  9.39393939,  9.6969697 , 10.        , 10.3030303 ,
       10.60606061, 10.90909091, 11.21212121, 11.51515152, 11.81818182,
       12.12121212, 12.42424242, 12.72727273, 13.03030303, 13.33333333,
       13.63636364, 13.93939394, 14.24242424, 14.54545455, 14.84848485,
       15.15151515, 15.45454545, 15.75757576, 16.06060606, 16.36363636,
       16.66666667, 16.96969697, 17.27272727, 17.57575758, 17.87878788,
       18.18181818, 18.48484848, 18.78787879, 19.09090909, 19.39393939,
       19.6969697 , 20.        , 20.3030303 , 20.60606061, 20.90909091,
       21.21212121, 21.51515152, 21.81818182, 22.12121212, 22.42424242,
       22.72727273, 23.03030303, 23.33333333, 23.63636364, 23.93939394,
       24.24242424, 24.54545455, 24.84848485, 25.15151515, 25.45454545,
       25.75757576, 26.06060606, 26.36363636, 26.66666667, 26.96969697,
       27.27272727, 27.57575758, 27.87878788, 28.18181818, 28.48484848,
       28.78787879, 29.09090909, 29.39393939, 29.6969697 , 30.        ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-cb13c70b-da52-45ce-b8d4-58a043cb4907' class='xr-section-summary-in' type='checkbox'  ><label for='section-cb13c70b-da52-45ce-b8d4-58a043cb4907' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>t</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-16a7c8b3-aef4-49fd-952b-96df5c7476d7' class='xr-index-data-in' type='checkbox'/><label for='index-16a7c8b3-aef4-49fd-952b-96df5c7476d7' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([                0.0, 0.10101010101010101, 0.20202020202020202,
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
      dtype=&#x27;float64&#x27;, name=&#x27;t&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-7d4c8c3c-c613-49ff-ae1a-2ddc2192b26c' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-7d4c8c3c-c613-49ff-ae1a-2ddc2192b26c' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



This returns a dataset which is of the exact same shape as the observation dataset, plus intermediate variables that were created during the simulation, if they are tracked by the solver.

Although this API seems to be a bit clunky, it is necessary, to make sure that simulations that are executed in parallel are isolated from each other.


## Estimating parameters 

We are almost set to infer the parameters of the model. We add another parameter to also estimate the error of the parameters, We use a lognormal distribution for it. We also specify an error model for the distribution. This will be 

$$y_{obs} \sim Normal (y, \sigma_y)$$

Further we also have to make some assumptions for the parameter $b$ which we want to fit. First, we have to define the prior function from which we draw the parameter values during the parameter estimation. Additionally, we set the `min` and `max` values for our parameters. This can also be done in one step,  as can be seen for the error-model parameter `sigma_y`.


```python
sim.config.model_parameters.b.prior = "lognorm(scale=1,s=1)"
sim.config.model_parameters.b.min = -5
sim.config.model_parameters.b.max = 5

#construction the error model
sim.config.model_parameters.sigma_y = Param(free=True , prior="lognorm(scale=1,s=1)", min=0, max=1)

sim.config.error_model.data = "normal(loc=data,scale=sigma_y)"
```

As `sigma_y` is not a fixed parameter, the new parameter does not have to be passed to the simulation class.


```python
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
sim.model_parameters['parameters']
```




    {'a': array(0), 'b': array(3), 'sigma_y': 0.0}



### Manual estimation

First, we try estimating the parameters by hand. For this we have a simple interactive backend. <br>
Note that changing sigma_y has no effect on the model fit because sigma_y is only used for the error model.


```python
from matplotlib import pyplot as plt
def plot(results: xr.Dataset):
    obs = sim.observations

    SSE = ((results.data - obs.data) ** 2).sum(dim="t") #calculating the sum of squared errors

    fig, ax = plt.subplots(1,1)
    ax.plot(results.t, results.data, lw=2, color="black")
    ax.plot(obs.t, obs.data, ls="", marker="o", color="tab:blue", alpha=.5)
    ax.set_xlim(-1,12)
    ax.set_ylim(-1,30)
    ax.text(0.05, 0.95, f"SSE={np.round(SSE.values, 2)}", transform=ax.transAxes, ha="left", va="top")
```


```python
sim.plot = plot
sim.interactive()
```


    HBox(children=(VBox(children=(FloatSlider(value=3.0, description='b', max=5.0, min=-5.0, step=None), FloatSlid…


### Estimating parameters and uncertainty with MCMC

Of course this example is very simple, we can in fact optimize the parameters perfectly by hand. But just for the fun of it, let's use *Markov Chain Monte Carlo* (MCMC) to estimate the parameters, their uncertainty and the uncertainty in the data. <br>

The inferer serves as the parameter estimator. Different inferer are implemented in numpy and can be found at the beginning of the tuorial and in the API. The method for the parameter estimation is defined by using {meth}`pymob.simulation.SimulationBase.set_inferer()`. This automatically translates the pymob data in the format of the selected inferer. Numpyro additionally needs a kernel. To start the estimation you use {meth}`pymob.simulation.SimulationBase.inferer.run()`.


*Note that other methods often don't need a kernel.*


```{admonition} numpyro distributions
:class: warning
Currently only few distributions are implemented in the numpyro backend. This API will soon change, so that basically any distribution can be used to specifcy parameters. 
```

Finally, we let our inferer run the paramter estimation procedure with the numpyro backend and a NUTS kernel. This does the job in a few seconds. <br>



```python
sim.dispatch_constructor() # important to call this before running the inferer

sim.set_inferer("numpyro")
sim.inferer.config.inference_numpyro.kernel = "nuts"
sim.inferer.run()

sim.inferer.idata.posterior
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
    data_obs dist 100 |
            value 100 |


      0%|                                                                                                                                                                  | 0/3000 [00:00<?, ?it/s]

    warmup:   0%|                                                                                                        | 1/3000 [00:00<43:42,  1.14it/s, 1 steps of size 1.87e+00. acc. prob=0.00]

    sample:  35%|██████████████████████████████████▏                                                                | 1036/3000 [00:00<00:01, 1454.27it/s, 7 steps of size 7.21e-01. acc. prob=0.95]

    sample:  70%|█████████████████████████████████████████████████████████████████████▏                             | 2096/3000 [00:01<00:00, 2982.18it/s, 1 steps of size 7.21e-01. acc. prob=0.94]

    sample: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:01<00:00, 2605.40it/s, 3 steps of size 7.21e-01. acc. prob=0.95]

    


    
                    mean       std    median      5.0%     95.0%     n_eff     r_hat
             b      2.65      0.03      2.65      2.61      2.70   1587.19      1.00
       sigma_y      1.55      0.12      1.55      1.36      1.75   1034.30      1.00
    
    Number of divergences: 0





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
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
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
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
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
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
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
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
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
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
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
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
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
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (chain: 1, draw: 2000)
Coordinates:
  * chain    (chain) int64 0
  * draw     (draw) int64 0 1 2 3 4 5 6 7 ... 1993 1994 1995 1996 1997 1998 1999
    cluster  (chain) int64 0
Data variables:
    b        (chain, draw) float32 2.703 2.623 2.604 2.64 ... 2.631 2.639 2.624
    sigma_y  (chain, draw) float32 1.475 1.762 1.667 1.612 ... 1.401 1.75 1.531
Attributes:
    created_at:     2025-10-10T13:25:53.558639+00:00
    arviz_version:  0.21.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-87f6da2a-68be-41f6-9215-f7dd5a1d3b80' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-87f6da2a-68be-41f6-9215-f7dd5a1d3b80' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 2000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-82eb51eb-343a-4af7-88d5-3cbf92923690' class='xr-section-summary-in' type='checkbox'  checked><label for='section-82eb51eb-343a-4af7-88d5-3cbf92923690' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-348c7c88-d9dc-4b1d-a999-926cb802cbca' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-348c7c88-d9dc-4b1d-a999-926cb802cbca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ae614a02-2c98-4a4e-a8b0-c4be3350e82e' class='xr-var-data-in' type='checkbox'><label for='data-ae614a02-2c98-4a4e-a8b0-c4be3350e82e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1996 1997 1998 1999</div><input id='attrs-18ebbc5a-f7eb-44aa-8d43-17d9e75941ef' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-18ebbc5a-f7eb-44aa-8d43-17d9e75941ef' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0fdebf39-d109-4341-95c3-b3480d1be255' class='xr-var-data-in' type='checkbox'><label for='data-0fdebf39-d109-4341-95c3-b3480d1be255' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1997, 1998, 1999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>cluster</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-e4436edb-689a-4ffc-a8a6-c35028f83e92' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e4436edb-689a-4ffc-a8a6-c35028f83e92' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-58324bac-cddd-4995-a0d6-044d1d5139e6' class='xr-var-data-in' type='checkbox'><label for='data-58324bac-cddd-4995-a0d6-044d1d5139e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-938b5f8e-8988-4858-962d-1506484ef077' class='xr-section-summary-in' type='checkbox'  checked><label for='section-938b5f8e-8988-4858-962d-1506484ef077' class='xr-section-summary' >Data variables: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>b</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>2.703 2.623 2.604 ... 2.639 2.624</div><input id='attrs-220eae5b-ad96-4240-b1f6-0df72985c3b0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-220eae5b-ad96-4240-b1f6-0df72985c3b0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-543ea8df-2569-4052-ab27-ea5dc3a96253' class='xr-var-data-in' type='checkbox'><label for='data-543ea8df-2569-4052-ab27-ea5dc3a96253' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[2.7032015, 2.6226218, 2.6036086, ..., 2.631157 , 2.6394734,
        2.6241918]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_y</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>1.475 1.762 1.667 ... 1.75 1.531</div><input id='attrs-f2155aae-2747-4890-96b2-ea2b34b7a6b7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f2155aae-2747-4890-96b2-ea2b34b7a6b7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ea6f08aa-04dd-4738-b1c9-996501132bc1' class='xr-var-data-in' type='checkbox'><label for='data-ea6f08aa-04dd-4738-b1c9-996501132bc1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.4750316, 1.7615653, 1.6671629, ..., 1.4010886, 1.7495875,
        1.5313449]], dtype=float32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c13cde2a-77dd-48fb-bd27-ae81de524023' class='xr-section-summary-in' type='checkbox'  ><label for='section-c13cde2a-77dd-48fb-bd27-ae81de524023' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-600eb6f4-4f5e-4568-88fc-c10973f6746c' class='xr-index-data-in' type='checkbox'/><label for='index-600eb6f4-4f5e-4568-88fc-c10973f6746c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-383080c5-4bc2-4d8e-bc94-24ebefb46d76' class='xr-index-data-in' type='checkbox'/><label for='index-383080c5-4bc2-4d8e-bc94-24ebefb46d76' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
       ...
       1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=2000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-eb8fa0f0-1e51-47e5-89e5-c213276307ad' class='xr-section-summary-in' type='checkbox'  checked><label for='section-eb8fa0f0-1e51-47e5-89e5-c213276307ad' class='xr-section-summary' >Attributes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2025-10-10T13:25:53.558639+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.21.0</dd></dl></div></li></ul></div></div>



We can inspect our estimates and see that the parameters are well esimtated by the model. Note that we only get an estimate for `b`. This is because earlier we set the parameter `a` with the flag `free=False` this effectively excludes it from estimation and uses the default value, which was set to the true value `a=0`. <br>

The `mean`of `b` is the value of the estimated parameter. It should be the same or close to estimation you did manually. The `sigma_y` is the mean error of this estimation.

### Plot the results

Pymob provides a very basic utility for plotting posterior predictions. We see that the mean is a perfect fit and also that the uncertainty in the data is correctly displayed. Fantstic 🎉


```python
sim.config.simulation.x_dimension = "t"
sim.posterior_predictive_checks(pred_hdi_style={"alpha": 0.1})
```


    
![png](Introduction_files/Introduction_42_0.png)
    



```{admonition} Customize the posterior predictive checks
:class: hint
You can explore the API of {class}`pymob.sim.plot.SimulationPlot` to find out how you can work on the default predictions. Of course you can always make your own plot, by accessing {attr}`pymob.simulation.inferer.idata` and {attr}`pymob.simulation.observations`
```

### Report the results
The command {meth}`pymob.simulation.SimulationBase.report()` can be used to generate an automated report. The report can be configured with options in {meth}`pymob.simulation.SimulationBase.config.report()`.


```python
sim.report()
```

    /export/home/fschunck/miniconda3/envs/pymob/lib/python3.11/site-packages/pymob/sim/report.py:230: UserWarning: There was an error compiling the report! Pandoc seems not to be installed. Make sure to install pandoc on your system. Install with: `conda install -c conda-forge pandoc` (https://pandoc.org/installing.html)
      warnings.warn(



## Exporting the simulation and running it via the case study API

After constructing the simulation, all settings of the simulation can be exported to a comprehensive configuration file, along with all the default settings. This is as simple as 


```python
import os
sim.config.case_study.name = "quickstart"
sim.config.case_study.scenario = "test"
sim.config.create_directory("scenario", force=True)
sim.config.create_directory("results", force=True)

# usually we expect to have a data directory in the case
os.makedirs(sim.data_path, exist_ok=True)
sim.save_observations(force=True)
sim.config.save(force=True)
```

    Scenario directory exists at '/export/home/fschunck/projects/pymob/docs/source/user_guide/case_studies/quickstart/scenarios/test'.
    Results directory exists at '/export/home/fschunck/projects/pymob/docs/source/user_guide/case_studies/quickstart/results/test'.


The simulation will be saved to the default path (`CASE_STUDY/scenarios/SCENARIO/settings.cfg`) or to a custom file path specified with the `fp` keyword. `force=True` will overwrite any existing config file, which is the reasonable choice in most cases.

From there on, the simulation is (almost) ready to be executable from the commandline.

### Commandline API

The commandline API runs a series of commands that load the case study, execute the {meth}`pymob.simulation.SimulationBase.initialize` method and perform some more initialization tasks, before running the required job.

+ `pymob-infer`: Runs an inference job e.g. `pymob-infer --case_study=quickstart --scenario=test --inference_backend=numpyro`. While there are more commandline options, these are the two required 

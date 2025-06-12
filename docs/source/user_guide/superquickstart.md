# Pymob quickstart

This super-quick quickstart provides an introduction to the basic Pymob workflow and its key functionalities.  
We will explore a simple linear regression model that we want to fit to a noisy dataset.  
Pymob supports the modeling process by providing several tools for *data structuring*, *parameter estimation* and *visualization of results*.  
  
If you are looking for a more detailed introduction, [click here]().  
If you want to learn how to work with ODE models, check out [this tutorial]().

## Pymob components

Before starting the modeling process, let's take a look at the main steps and modules of pymob:

1. __Simulation:__   
First, we need to initialize a Simulation object by creating an instance of the {class}`pymob.simulation.SimulationBase` class from the simulation module.   
Optionally, we can configure the simulation with `sim.config.case_study.name = "linear-regression"`, `sim.config.case_study.scenario = "test"` and many other options. 

2. __Model:__   
Our model will be defined as a standard python function.  
We will then assign it to the Simulation object by accessing the `.model` attribute. 

3. __Observations:__   
Our observation data must be structured as a [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).  
We assign it to the `.observations` attribute of our Simulation object.   
Calling `sim.config.data_structure` will give us further information about the layout of our data.  

4. __Solver:__  
A [solver](https://pymob.readthedocs.io/en/stable/api/pymob.solvers.html) is required to solve the model.   
In our simple case, we will use the `solve_analytic_1d` solver from the `pymob.solver.analytic` module.  
We assign it to our Simulation object using the `.solver` attribute.   
Since our model already provides an analytical solution, this solver basically does nothing. It is still needed to fulfill Pymob's requirement for a solver component.   
For more complex models (e.g. ODEs), the `JaxSolver` from the `pymob.solvers.diffrax` module is a more powerful option.   
Users can also implement custom solvers as a subclass of `pymob.solver.SolverBase`.   
  
5. __Inferer:__  
The inferer handels the parameter estimation.  
Pymob supports [various backends](https://pymob.readthedocs.io/en/stable/user_guide/framework_overview.html). In this example, we will work with *NumPyro*.  
We assign the inferer to our Simulation object via the `.inferer` attribute and configure the desired kernel (e.g. *nuts*).  
But before inference, we need to parameterize our model using the *Param* class.   
Each parameter can be marked either as free or fixed, depending on whether it should be variable during the optimization procedure.   
The parameters are stored in the `sim.model_parameters` dictionary, which holds model input values.
By default, it takes the keys: `parameters`, `y0` and `x_in`. 

6. __Evaluator:__  
The Evaluator is an instance to manage model evaluations.
It sets up tasks, coordinates parallel runs of the simulation and keeps track of the results from each simulation or parameter inference process.  

7. __Config:__  
The simulation settings will be saved in a `.cfg` configuration file.  
The config file contains information about our simulation in various sections. -> [Learn more here](https://pymob.readthedocs.io/en/stable/user_guide/case_studies.html#configuration).  
We can further use it to create new simulations by loading settings from a config file. 


![framework-overview](.\figures\pymob_overview.png)


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
$y_{obs}$ represents the observed data over the time $t$ [0, 10].  
To use this data later in the simulation, we need to convert it into a xarray-Dataset.  
In your own application, you would replace this with your measured experimental data.  


```python
# Parameter for the artificial data generation
slope = np.random.uniform(2.0, 4.0) 
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
    y        (t) float64 2.537 0.762 3.105 3.813 ... 33.87 34.32 37.92 39.47</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-3cba92b8-0944-42c5-b82a-35783b953cd4' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-3cba92b8-0944-42c5-b82a-35783b953cd4' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>t</span>: 100</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-a797f301-1248-4646-9731-91e28642f5e8' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a797f301-1248-4646-9731-91e28642f5e8' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>t</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.101 0.202 ... 9.899 10.0</div><input id='attrs-c80e20c8-6a9d-4f7e-9478-466ed8adac53' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c80e20c8-6a9d-4f7e-9478-466ed8adac53' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-60347dee-7fd7-40d1-b503-b9cd31e0334a' class='xr-var-data-in' type='checkbox'><label for='data-60347dee-7fd7-40d1-b503-b9cd31e0334a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.      ,  0.10101 ,  0.20202 ,  0.30303 ,  0.40404 ,  0.505051,
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
        9.69697 ,  9.79798 ,  9.89899 , 10.      ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-73cae99e-8b2f-4feb-824c-55817b00ee92' class='xr-section-summary-in' type='checkbox'  checked><label for='section-73cae99e-8b2f-4feb-824c-55817b00ee92' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.537 0.762 3.105 ... 37.92 39.47</div><input id='attrs-44ebf413-4dc2-4ed5-b79b-4a22c6c5d3c2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-44ebf413-4dc2-4ed5-b79b-4a22c6c5d3c2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c8abb7e6-b7f3-4dfa-914b-0afef1078d02' class='xr-var-data-in' type='checkbox'><label for='data-c8abb7e6-b7f3-4dfa-914b-0afef1078d02' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 2.53735306,  0.76202974,  3.10502331,  3.81269502,  1.65493898,
        4.7382624 ,  1.91659115,  5.09948271,  4.09436031,  2.93203812,
        2.41660779,  3.52721064,  4.52038892,  5.25244692,  6.49443682,
        7.15817191,  7.37549405,  6.45793422, 10.42760157,  7.0160899 ,
        8.44228028,  8.14150763,  8.33164821,  9.06734271, 10.1364285 ,
        9.89235673, 11.86570089,  8.92148715,  8.00979706, 12.612303  ,
       12.45222116, 14.50730789, 11.04216221, 13.04196904, 10.78455757,
       14.51374821, 15.78655397, 11.85873628, 15.17499665, 12.55360977,
       17.20997672, 14.13169503, 13.91247417, 17.84122887, 19.53174081,
       18.97160079, 20.69099192, 19.09570155, 19.80465631, 21.57411683,
       18.67932246, 15.86782202, 21.89297698, 21.14301831, 20.60363498,
       23.19466707, 21.78505986, 24.02588231, 19.93938713, 24.11449125,
       23.38553814, 20.22692378, 22.04672104, 24.69358621, 26.35163201,
       26.87845261, 25.66642845, 24.78028588, 28.13555625, 28.77362647,
       29.8394705 , 28.96746404, 26.8770062 , 28.38000956, 27.32974723,
       28.48364593, 28.19300058, 31.5562261 , 27.70039454, 30.37674089,
       33.23517464, 31.15096457, 31.92976207, 30.66092298, 37.31956867,
       33.85527604, 30.32904845, 31.79863742, 32.96114687, 35.05260485,
       36.40587357, 32.88890392, 33.20810035, 35.01590153, 35.25401727,
       39.29738961, 33.86973557, 34.3242026 , 37.91957645, 39.46912001])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1bb34202-8817-4d17-b85c-9a071e56812b' class='xr-section-summary-in' type='checkbox'  ><label for='section-1bb34202-8817-4d17-b85c-9a071e56812b' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>t</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-f2ac7879-0c60-4f2e-8504-7516258bff90' class='xr-index-data-in' type='checkbox'/><label for='index-f2ac7879-0c60-4f2e-8504-7516258bff90' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([                0.0, 0.10101010101010101, 0.20202020202020202,
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
      dtype=&#x27;float64&#x27;, name=&#x27;t&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-89be94e4-c853-4f55-9cc5-db089c3863c0' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-89be94e4-c853-4f55-9cc5-db089c3863c0' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




    
![png](superquickstart_files/superquickstart_6_1.png)
    


## Initialize a simulation

In pymob, a Simulation object is initialized by creating an instance of the {class}`pymob.simulation.SimulationBase` class from the simulation module.  
We will choose a linear regression model, as it provides a good approximation of the data: $ y = a + b*x $

```{admonition} x-dimension
:class: note
The x_dimension of our simulation can have any name, for example t as often used for time series data.
You can specify it via `sim.config.simulation.x_dimension`.
```


```python
# Initialize the Simulation object
sim = SimulationBase()

# Define the linear regression model
def linreg(x, a, b):
    return a + b * x

# Add the model to the simulation
sim.model = linreg

# Adding our dataset to the simulation
sim.observations = data_obs

# Defining a solver
sim.solver = solve_analytic_1d
```

    MinMaxScaler(variable=y, min=0.7620297399871993, max=39.46912001079589)
    

    C:\Users\mgrho\pymob\pymob\simulation.py:303: UserWarning: `sim.config.data_structure.y = Datavariable(dimensions=['t'] min=0.7620297399871993 max=39.46912001079589 observed=True dimensions_evaluator=None)` has been assumed from `sim.observations`. If the order of the dimensions should be different, specify `sim.config.data_structure.y = DataVariable(dimensions=[...], ...)` manually.
      warnings.warn(
    

```{admonition} Scalers
:class: note
We notice a mysterious Scaler message. This tells us that our data variable has been identified and a scaler was constructed, which transforms the variable between [0, 1].   
This has no effect at the moment, but it can be used later. Scaling can be powerful to help parameter estimation in more complex models.
```


## Running the model 🏃

Next, we define the model parameters *a* and *b*.  
Parameter *a* is set as fixed (`free = False`), meaning its value is known and will not be estimated during optimization.  
Parameter *b* is marked as free (`free = True`), allowing it to be optimized to fit the data. As an initial guess, we assume b = 3.   

Our model is now prepared with a defined parameter set.  
To initialize the *Evaluator* class, we call `sim.dispatch_constructor()`.   
This step is essential and must be executed every time changes are made to the model. 

The returned dataset (`evaluator.results`) has the exact same shape as the observation data.


```python
# Parameterizing the model
sim.config.model_parameters.a = Param(value=1, free=False)
sim.config.model_parameters.b = Param(value=3, free=True)
# this makes sure the model parameters are available to the model.
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

# put everything in place for running the simulation
sim.dispatch_constructor()

# run
evaluator = sim.dispatch(theta={"b":3})
evaluator()
evaluator.results
```

    C:\Users\mgrho\pymob\pymob\simulation.py:552: UserWarning: The number of ODE states was not specified in the config file [simulation] > 'n_ode_states = <n>'. Extracted the return arguments ['a+b*x'] from the source code. Setting 'n_ode_states=1.
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
    y        (t) float64 1.0 1.303 1.606 1.909 2.212 ... 30.09 30.39 30.7 31.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-1c65b503-5085-41c4-8541-11fd2419ad56' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-1c65b503-5085-41c4-8541-11fd2419ad56' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>t</span>: 100</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-5019ed9d-7d01-441b-bef4-dfd10f6916b8' class='xr-section-summary-in' type='checkbox'  checked><label for='section-5019ed9d-7d01-441b-bef4-dfd10f6916b8' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>t</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.0 0.101 0.202 ... 9.899 10.0</div><input id='attrs-3bd2447e-bca7-41d2-b649-383901267a04' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3bd2447e-bca7-41d2-b649-383901267a04' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cbd54d71-5413-4299-8c7b-d851d64cc0a2' class='xr-var-data-in' type='checkbox'><label for='data-cbd54d71-5413-4299-8c7b-d851d64cc0a2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0.      ,  0.10101 ,  0.20202 ,  0.30303 ,  0.40404 ,  0.505051,
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
        9.69697 ,  9.79798 ,  9.89899 , 10.      ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-d063be66-88cf-4f10-a8f4-1bcb5972929f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-d063be66-88cf-4f10-a8f4-1bcb5972929f' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(t)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.0 1.303 1.606 ... 30.39 30.7 31.0</div><input id='attrs-af1fff76-0563-4521-8e7c-7c457f5042e1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-af1fff76-0563-4521-8e7c-7c457f5042e1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f046512e-5cb8-499e-93c7-5827a840e920' class='xr-var-data-in' type='checkbox'><label for='data-f046512e-5cb8-499e-93c7-5827a840e920' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 1.        ,  1.3030303 ,  1.60606061,  1.90909091,  2.21212121,
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
       29.78787879, 30.09090909, 30.39393939, 30.6969697 , 31.        ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-205c757b-6e11-466d-8fab-67d9291fac41' class='xr-section-summary-in' type='checkbox'  ><label for='section-205c757b-6e11-466d-8fab-67d9291fac41' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>t</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-389aa368-2271-4700-81bd-a9e5b3a74441' class='xr-index-data-in' type='checkbox'/><label for='index-389aa368-2271-4700-81bd-a9e5b3a74441' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([                0.0, 0.10101010101010101, 0.20202020202020202,
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
      dtype=&#x27;float64&#x27;, name=&#x27;t&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-cd67cdd0-d69e-4aab-b3c9-91c81ac9fe9b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cd67cdd0-d69e-4aab-b3c9-91c81ac9fe9b' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Let's take a look at the results.  

You can vary the parameter *b* in the previous step to investigate its influence on the model fit.  
In the [Beginner Guide](), you can try out the *manual parameter estimation*, which is a feature provided by Pymob.  


```python
fig, ax = plt.subplots(figsize=(5, 4))
data_res = evaluator.results
ax.plot(data_obs.t, data_obs.y, ls="", marker="o", color="tab:blue", alpha=.5, label ="observation data")
ax.plot(data_res.t, data_res.y, color="black", label ="result")
ax.legend()
```




    <matplotlib.legend.Legend at 0x25f8941ebd0>




    
![png](superquickstart_files/superquickstart_14_1.png)
    


## Estimating parameters and uncertainty with MCMC
Of course this example is very simple - we could, in fact, optimize the parameters perfectly by hand. But just for fun, let's use *Markov Chain Monte Carlo (MCMC)* to estimate the parameters, their uncertainty and the uncertainty in the data. We’ll run the parameter estimation with our inferer, using the NumPyro backend with a NUTS kernel. This completes the job in a few seconds.

We are almost ready to infer the model parameters. To also estimate the uncertainty of the parameters, we add another parameter representing the error and assume that it follows a lognormal distribution. Additionally, we specify an error model for the data distribution. This will be: $$y_{obs} \sim Normal (y, \sigma_y)$$  


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

    C:\Users\mgrho\pymob\pymob\inference\numpyro_backend.py:552: UserWarning: Model is not rendered, because the graphviz executable is not found. Try search for 'graphviz executables not found' and the used OS. This should be an easy fix :-)
      warnings.warn(
    

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
    

    sample: 100%|██████████| 3000/3000 [00:02<00:00, 1240.00it/s, 3 steps of size 7.01e-01. acc. prob=0.94]
    

    
                    mean       std    median      5.0%     95.0%     n_eff     r_hat
             b      3.68      0.03      3.68      3.63      3.73   1376.15      1.00
       sigma_y      1.75      0.13      1.74      1.54      1.97   1188.08      1.00
    
    Number of divergences: 0
    


    
![png](superquickstart_files/superquickstart_16_4.png)
    


```{admonition} numpyro distributions
:class: warning
Currently only few distributions are implemented in the numpyro backend. This API will soon change, so that basically any distribution can be used to specifcy parameters. 
```

We can inspect our estimates and ssee that the model provides a good fit for the parameters.  
Note that we only get an estimate for $b$. Previously, we set the parameter $a$ with the flag `free = False`.   
This effectively excludes it from the estimation and uses its default value, which was set to the true value `a = 0`.


```{admonition} Customize the posterior predictive checks
:class: hint
You can explore the API of {class}`pymob.sim.plot.SimulationPlot` to find out how you can work on the default predictions. Of course you can always make your own plot, by accessing {attr}`pymob.simulation.inferer.idata` and {attr}`pymob.simulation.observations`
```

## Report the results

Pymob provides an option to generate an automated report of the parameter distribution.  
The report can be configured by modifying the options in `sim.config.report`.


```python
# report the results
sim.report()
```


    
![png](superquickstart_files/superquickstart_21_0.png)
    



    
![png](superquickstart_files/superquickstart_21_1.png)
    



## Exporting the simulation and running it via the case study API

After constructing the simulation, all settings - custom and default - can be exported to a comprehensive configuration file.   
The simulation will be saved to the default path (`CASE_STUDY/scenarios/SCENARIO/settings.cfg`) or to a custom path, specified with the file path `fp` keyword.   
Setting `force=True` will overwrite any existing config file, which is a reasonable choice in most cases.
From this point on, the simulation is (almost) ready to be executed from the command-line. 


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

    Scenario directory exists at 'c:\Users\mgrho\pymob\docs\source\user_guide\case_studies\quickstart\scenarios\test'.
    Results directory exists at 'c:\Users\mgrho\pymob\docs\source\user_guide\case_studies\quickstart\results\test'.
    

### Commandline API

The command-line API runs a series of commands that load the case study, execute the {meth}`pymob.simulation.SimulationBase.initialize` method and perform some more initialization tasks before running the required job.

+ `pymob-infer` runs an inference job, for example:  

  `pymob-infer --case_study=quickstart --scenario=test --inference_backend=numpyro`.   
  While there are more command-line options, these two (--case_study and --scenario) are required.



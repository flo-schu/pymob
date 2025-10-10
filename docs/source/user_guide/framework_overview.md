# Framework overview

Pymob is built around {class}`pymob.simulation.SimulationBase`, which is the object where all necessary information are pooled. For configuration, `pymob` relies on {class}`pymob.sim.config.Config`, which uses [`pydantic`](https://docs.pydantic.dev/latest/) to validate the configuration, before it is used to set up the simulation. 


```python
from pymob import SimulationBase

# initializing a Simulation with a config file
sim = SimulationBase()

# accessing the config file
sim.config
```




    Config(case_study=Casestudy(init_root='/export/home/fschunck/projects/pymob/docs/source/user_guide', root='.', name='unnamed_case_study', version=None, pymob_version='0.5.19', scenario='unnamed_scenario', package='case_studies', modules=['sim', 'mod', 'prob', 'data', 'plot'], simulation='Simulation', output=None, data=None, scenario_path_override=None, observations=None, logging='DEBUG', logfile=None, output_path='case_studies/unnamed_case_study/results/unnamed_scenario', data_path='case_studies/unnamed_case_study/data', default_settings_path='case_studies/unnamed_case_study/scenarios/unnamed_scenario/settings.cfg'), simulation=Simulation(model=None, solver=None, y0=[], x_in=[], input_files=[], n_ode_states=-1, batch_dimension='batch_id', x_dimension='time', modeltype='deterministic', solver_post_processing=None, seed=1), data_structure=Datastructure(), solverbase=Solverbase(x_dim='time', exclude_kwargs_model=('t', 'time', 'x_in', 'y', 'x', 'Y', 'X'), exclude_kwargs_postprocessing=('t', 'time', 'interpolation', 'results')), jaxsolver=Jaxsolver(diffrax_solver='Dopri5', rtol=1e-06, atol=1e-07, pcoeff=0.0, icoeff=1.0, dcoeff=0.0, max_steps=100000, throw_exception=True), inference=Inference(eps=1e-08, objective_function='total_average', n_objectives=1, objective_names=[], backend=None, extra_vars=[], plot=None, n_predictions=100), model_parameters=Modelparameters(), error_model=Errormodel(), multiprocessing=Multiprocessing(cores=1), inference_pyabc=Pyabc(sampler='SingleCoreSampler', population_size=100, minimum_epsilon=0.0, min_eps_diff=0.0, max_nr_populations=1000, database_path='/tmp/pyabc.db'), inference_pyabc_redis=Redis(password='nopassword', port=1111, n_predictions=50, history_id=-1, model_id=0), inference_pymoo=Pymoo(algortihm='UNSGA3', population_size=100, max_nr_populations=1000, ftol=1e-05, xtol=1e-07, cvtol=1e-07, verbose=True), inference_numpyro=Numpyro(user_defined_probability_model=None, user_defined_error_model=None, user_defined_preprocessing=None, gaussian_base_distribution=False, kernel='nuts', init_strategy='init_to_uniform', chains=1, draws=2000, warmup=1000, thinning=1, nuts_draws=2000, nuts_step_size=0.8, nuts_max_tree_depth=10, nuts_target_accept_prob=0.8, nuts_dense_mass=True, nuts_adapt_step_size=True, nuts_adapt_mass_matrix=True, sa_adapt_state_size=None, svi_iterations=10000, svi_learning_rate=0.0001), report=Report(debug_report=False, pandoc_output_format='html', model=True, parameters=True, parameters_format='pandas', diagnostics=True, diagnostics_with_batch_dim_vars=False, diagnostics_exclude_vars=[], goodness_of_fit=True, goodness_of_fit_use_predictions=True, goodness_of_fit_nrmse_mode='range', table_parameter_estimates=True, table_parameter_estimates_format='csv', table_parameter_estimates_significant_figures=3, table_parameter_estimates_error_metric='sd', table_parameter_estimates_parameters_as_rows=True, table_parameter_estimates_with_batch_dim_vars=False, table_parameter_estimates_exclude_vars=[], table_parameter_estimates_override_names={}, plot_trace=True, plot_parameter_pairs=True))



## Pymob API

![framework-overview](./figures/pymob_overview.png)

### Pymob exposes the following input and output interfaces

#### Config

Pymob uses [`pydantic`](https://docs.pydantic.dev/latest/) Models for validation of the configuration files. The configuration is organized into sections, e.g.


```python
sim.config.data_structure
```




    Datastructure()



We use a {class}`~pymob.sim.config.DataVariable` to populate the data structure. Let's say we have made some concentration measurments


```python
from pymob.sim.config import DataVariable

sim.config.data_structure.y = DataVariable(dimensions=["x"], observed=True)
sim.config.data_structure
```




    Datastructure(y=DataVariable(dimensions=['x'], min=nan, max=nan, observed=True, dimensions_evaluator=None))



Configurations can be changed in the files before a simulation is initialized from a config file, or directly in the script.


```python
sim.config.data_structure.y.min = 0
sim.config.data_structure
```




    Datastructure(y=DataVariable(dimensions=['x'], min=0.0, max=nan, observed=True, dimensions_evaluator=None))



As can be seen in the figure above, it is the communication between Simulation class and config files is bidirectional, this means, Simulations can be created from config files or in a scripting environment, and successively exported to config files. For more information see [configuration](case_studies.md#configuration) for details

#### Solver

Solvers solve the model. In order to automatize dimension handling and solving the model for the correct coordinates. Solvers subclass {class}`pymob.solver.SolverBase`.


```python
from pymob.solvers.analytic import solve_analytic_1d

sim.solver = solve_analytic_1d
```

#### Model

Models are provided as plain Python functions. 


```python
def linear_regression(x, a, b):
    return b * x + a

sim.model = linear_regression
```

#### Coordinates

A model is evaluated along the coordinates of the model dimensions


```python
import numpy as np
sim.config.simulation.x_dimension = "x"

sim.coordinates["x"] = np.array([0,1,2,3,4,5])
```

#### Model parameters

Model parameters carry any function parameters, initial values (for ODE models) and other model input such as forcings


#### .dispatch()

{func}`~pymob.simulation.SimulationBase.dispatch()` launches a forward pass through the simulation. {func}`~pymob.simulation.SimulationBase.dispatch_constructor()` does all the repetitive work that is only necessary once. It is recommended to use `dispatch_constructor` after any configuration change


```python
sim.dispatch_constructor()

evaluator_1 = sim.dispatch({"a": 1, "b": 2})
evaluator_2 = sim.dispatch({"a": 10, "b": 3})

evaluator_1()
evaluator_2()
```

    /export/home/fschunck/miniconda3/envs/pymob/lib/python3.11/site-packages/pymob/simulation.py:706: UserWarning: The number of ODE states was not specified in the config file [simulation] > 'n_ode_states = <n>'. Extracted the return arguments ['b*x+a'] from the source code. Setting 'n_ode_states=1.
      warnings.warn(


#### Observations

Observations are required to be xarray Datasets. An [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) is a collection of annotated arrays, using HDF5 data formats for input/output operations.

#### Simulation results

Simulation results are returned by the solver. Plainly they are returned as dictionaries containing NDarrays. However, due to the information contained in the observations dataset, the results dictionary is automatically casted to an [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html), which has the same shape as the observations. This makes comparisons between observations and simulations extremely easy.


```python
evaluator_1.results
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
Dimensions:  (x: 6)
Coordinates:
  * x        (x) int64 0 1 2 3 4 5
Data variables:
    y        (x) int64 1 3 5 7 9 11</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-868eb53e-b0b9-4d81-86ff-318252b6092f' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-868eb53e-b0b9-4d81-86ff-318252b6092f' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 6</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-efb719dd-4985-4030-87a0-38df67a98abd' class='xr-section-summary-in' type='checkbox'  checked><label for='section-efb719dd-4985-4030-87a0-38df67a98abd' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5</div><input id='attrs-4e67974f-5c9e-4050-af09-f422afaa5f2c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4e67974f-5c9e-4050-af09-f422afaa5f2c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2f7fee5f-ebbb-4e4b-990c-9e86c67e1a06' class='xr-var-data-in' type='checkbox'><label for='data-2f7fee5f-ebbb-4e4b-990c-9e86c67e1a06' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-21c1e3f1-f313-4ba9-bfd6-3587aecb3965' class='xr-section-summary-in' type='checkbox'  checked><label for='section-21c1e3f1-f313-4ba9-bfd6-3587aecb3965' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1 3 5 7 9 11</div><input id='attrs-42cba53a-03ef-4632-b086-e7a364ef8d1c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-42cba53a-03ef-4632-b086-e7a364ef8d1c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-edaffb3f-748d-4e0e-8e88-482cef4a49ca' class='xr-var-data-in' type='checkbox'><label for='data-edaffb3f-748d-4e0e-8e88-482cef4a49ca' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 1,  3,  5,  7,  9, 11])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-42c11250-d6f2-45e0-83fe-bc1cfa143abe' class='xr-section-summary-in' type='checkbox'  ><label for='section-42c11250-d6f2-45e0-83fe-bc1cfa143abe' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>x</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-eeb7cbc8-dc9f-4acb-86d8-23ccac3ff2cd' class='xr-index-data-in' type='checkbox'/><label for='index-eeb7cbc8-dc9f-4acb-86d8-23ccac3ff2cd' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5], dtype=&#x27;int64&#x27;, name=&#x27;x&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8c03ad96-d5b2-43c1-8c32-0fb7d35ac8d7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-8c03ad96-d5b2-43c1-8c32-0fb7d35ac8d7' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
evaluator_2.results
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
Dimensions:  (x: 6)
Coordinates:
  * x        (x) int64 0 1 2 3 4 5
Data variables:
    y        (x) int64 10 13 16 19 22 25</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-89135716-df99-425e-ae52-1233485f65e3' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-89135716-df99-425e-ae52-1233485f65e3' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 6</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-35a6fa49-8072-4219-91bf-7e294866dfb6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-35a6fa49-8072-4219-91bf-7e294866dfb6' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5</div><input id='attrs-1dc7677e-ef16-4dde-bed4-db3f937fd4ed' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1dc7677e-ef16-4dde-bed4-db3f937fd4ed' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6603b1b5-8ff5-43aa-8752-e09a03666069' class='xr-var-data-in' type='checkbox'><label for='data-6603b1b5-8ff5-43aa-8752-e09a03666069' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-095ceee0-6f2b-4b88-8756-2897fafacddb' class='xr-section-summary-in' type='checkbox'  checked><label for='section-095ceee0-6f2b-4b88-8756-2897fafacddb' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>10 13 16 19 22 25</div><input id='attrs-668d9ed2-21e2-4f08-808a-2b7f5162d454' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-668d9ed2-21e2-4f08-808a-2b7f5162d454' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dc5a7b7c-e684-4e85-9cc6-f8ae3012d7ce' class='xr-var-data-in' type='checkbox'><label for='data-dc5a7b7c-e684-4e85-9cc6-f8ae3012d7ce' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([10, 13, 16, 19, 22, 25])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1dbf0755-e7e9-42de-ba14-f300d63846b6' class='xr-section-summary-in' type='checkbox'  ><label for='section-1dbf0755-e7e9-42de-ba14-f300d63846b6' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>x</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-a3ce0375-a96c-4110-884d-77eb8a8e33b1' class='xr-index-data-in' type='checkbox'/><label for='index-a3ce0375-a96c-4110-884d-77eb8a8e33b1' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5], dtype=&#x27;int64&#x27;, name=&#x27;x&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8584ca52-22e2-41ac-8d2e-ba86d39e4d71' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-8584ca52-22e2-41ac-8d2e-ba86d39e4d71' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



#### Parameter estimates

Parameter estimates are harmonized by reporting them as [`arviz.InferenceData`](https://python.arviz.org/en/latest/getting_started/WorkingWithInferenceData.html) using `xarray.Datasets` under the hood. Thereby `pymob` supports variably dimensional datasets


## Parameter estimation

Parameter estimation is implemented through backends, which can be seen as converters between the {class}`pymob.simulation.SimulationBase` object and the API of the Inference tool. Inference backends are selected by using 


```python
sim.set_inferer("numpyro")
```

    Jax 64 bit mode: False
    Absolute tolerance: 1e-07


### Supported Algorithms and Planned Features

| Backend | Supported Algorithms | Inference | Hierarchical Models |
| :--- | --- | --- | --- |
| `numpyro` | Markov Chain Monte Carlo (MCMC), Stochastic Variational Inference (SVI) | ✅ | ✅ |
| `pymoo` | (Global) Multi-objective optimization | ✅ | plan |
| `pyabc` | Approximate Bayes | ✅ | plan |
| `scipy` | Local optimization (`minimize`) | dev | plan |
| `pymc` | MCMC | plan | plan |
| `sbi` | Simulation Based Inference (in planning) | hold | hold |
| `interactive ` | interactive backend in jupyter notebookswith parameter sliders | ✅ | plan |

---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for pymob, with links to the rest
      of the site..
html_theme.sidebar_secondary.remove: true
---

Pymob documentation
===================


Pymob is a Python based model building platform. 
It automates repetitive tasks in the modeling process so that you can focus on building models, asking questions to the real world and learn from observations.

The idea of `pymob` originated from the frustration with fitting complex models to complicated datasets (missing observations, non-uniform data structure, non-linear models, ODE models). In such scenarios a lot of time is spent matching observations with model results.

Usually, parameter estimation/optimization/inference algorithms have dedicated requirements for how to input data and specify parameters. This produces a significant barrier to explore different, potentially better suited algorithms to estimate the data. 


## The main goals of `pymob` are

+ providing a **uniform interface** for describing models  
+ using this interface to **fit models to data** with a variety of optimization and inference algorithms
+ enabling **reproducibility** by versioning case studies and distributing them as python packages

### Supported Algorithms and Planned Features

| Backend | Supported Algorithms | Inference | Hierarchical Models |
| :--- | --- | --- | --- |
| `numpyro` | Markov Chain Monte Carlo (MCMC), Stochastic Variational Inference (SVI) | ✅ | ✅ |
| `pymoo` | (Global) Multi-objective optimization | ✅ | plan |
| `pyabc` | Approximate Bayes | ✅ | plan |
| `scipy` | Local optimization (`minimize`) | dev | plan |
| `pymc` | MCMC | plan | plan
| `sbi` | Simulation Based Inference (in planning) | hold | hold


### Framework overview


![framework-overview](./user_guide/figures/pymob_overview.png)

#### Pymob exposes the following input and output interfaces

+ **solver**: Solvers solve the model. In order to automatize dimension handling and solving the model for the correct coordinates. Solvers subclass {class}`pymob.solver.SolverBase`. 
+ **model**: Models are provided as plain Python functions. 
+ **data**: [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) collections of annotated arrays, using HDF5 data formats for I/O 
+ **simulation results**: [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) following the same structure as observations
+ **parameter estimates**:Parameter estimates are harmonized by reporting them as [`arviz.InferenceData`](https://python.arviz.org/en/latest/getting_started/WorkingWithInferenceData.html) using `xarray.Datasets` under the hood. Thereby `pymob` supports variably dimensional datasets
+ **config**: [`pydantic`](https://docs.pydantic.dev/latest/) Models for validation of configuration files 



```{toctree}
:maxdepth: 1
:caption: Contents

User Guide <user_guide/index>
Examples <examples/index>
API reference <api/pymob>
```



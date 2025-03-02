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


## Next steps

```{toctree}
:maxdepth: 1
:caption: Contents

User Guide <user_guide/index>
Examples <examples/index>
API reference <api/pymob>
```



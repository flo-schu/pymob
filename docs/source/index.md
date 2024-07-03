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
It abstracts repetitive tasks in the modeling process so that you can focus on building models, asking questions to the real world and learn from observations.

The idea of `pymob` originated from the frustration with fitting complex models to complicated datasets (missing observations, non-uniform data structure, non-linear models, ODE models). In such scenarios a lot of time is spent matching observations with model results.

The main strength of `pymob` is to provide a uniform interface for describing models and using this model to fit a variety of state-of-the-art optimization and inference algorithms on it.


```{toctree}
:maxdepth: 1
:caption: Contents

User Guide <user_guide/index>
Examples <examples/index>
API reference <api/pymob>
```



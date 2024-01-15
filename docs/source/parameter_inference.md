# Parameter inference

Parameterizing complex models is still surprisingly difficult. it is both fortunate
that many different tools exist today to approach model parameterization. This task
has a variety of names with slightly different associated meaning, but they all
address the same task. Namely, estimating the set of parameters that best describe
some data $Y$ under the (mathematical) assumptions of the model. This process can be called parameter inference, parameter estimation, optimization, calibration, fitting. While there are
some subtle differences between the terms, they can be used more or less interchangibly. Relevant differences arise from the tools being used. The terms could be classified based on the probabilistic nature of the approach. While calibration and optimization are rather associated with deterministic models, which should have a unique optimal solution; estimation and inference are rather associated with approaches where the data are assumed to originate from a probability distribution.

For simplicity we will use the term parameter **inference**, because we interpret by its meaning from the fields of logic and reasoning, where it describes the process from moving from premises to logical consequences. 

> Statistical inference uses mathematics to draw conclusions in the presence of uncertainty. This generalizes deterministic reasoning, with the absence of uncertainty as a special case. Statistical inference uses quantitative or qualitative (categorical) data which may be subject to random variations. 

## Why use `pymob.inference`?

The goal of `pymob` is to reduce the overhead for exploring, parameterizing and comparing
models so that the focus can remain on model development. Because after all, 
we would like to use models to anwer questions.

`pymob.inference` only enforces minimal conventions, which allows the interoperability between different frameworks. At the same time it allows you to take the models you have already written and just plug them into the framework and use it to do inference on it. 

 + Observations/data have to be provided as one [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).
 + Simulations have to be wrapped inside a function, which converts the simulation output to a dictionary of results, mapping the results to so called data-variables

The changes to your model are minimal and there are tools provided to prepare your simulator and your data for the task.

```{admonition} TODO
:class: attention
- [ ] Create IO tools for importing existing models and data 
- [ ] Write documentation with examples
```
.. pymob documentation master file, created by
   sphinx-quickstart on Mon Jan 15 11:21:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pymob's documentation!
=================================

Pymob is a Python based model building platform. 
It abstracts repetitive tasks in the modeling process so that you can focus on building models, asking questions to the real world and learn from observations.

The idea of `pymob` originated from the frustration with fitting complex models to complicated datasets (missing observations, non-uniform data structure, non-linear models, ODE models). In such scenarios a lot of time is spent matching observations with model results.

The main strength of `pymob` is to provide a uniform interface for describing models and using this model to fit a variety of state-of-the-art optimization and inference algorithms on it.

Currently, supported inference backends are:

* interactive (interactive backend in jupyter notebookswith parameter sliders)
* numpyro (bayesian inference and stochastic variational inference)
* pyabc (approximate bayesian inference)
* pymoo (experimental! multi-objective optimization)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   case_studies
   simulation
   parameter_inference

   api/pymob


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

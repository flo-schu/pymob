# Hierarchical Predator Prey modelling

The Lotka-Volterra predator-prey model is the archetypical model for dynamical systems, depicting the fluctuating population development of the dynamical system. 
It is simple enough to fit parameters and estimate their uncertainty in a single replicate. But what if there was some environmental fluctuation we wanted 


```python
import numpy as np
import arviz as az
import xarray as xr
import matplotlib.pyplot as plt

from pymob import Config
from pymob.inference.scipy_backend import ScipyBackend
from pymob.sim.parameters import Param
from pymob.sim.config import Modelparameters
from pymob.solvers.diffrax import JaxSolver
from pymob.inference.analysis import plot_pair
```


```python
# import case study and simulation

config = Config()
config.case_study.package = "../.."
config.case_study.name = "lotka_volterra_case_study"
config.case_study.scenario = "test_hierarchical"
config.case_study.simulation = "HierarchicalSimulation"
config.import_casestudy_modules(reset_path=True)

from lotka_volterra_case_study.sim import HierarchicalSimulation

sim = HierarchicalSimulation(config)
sim.initialize_from_script()

```

## Setting up the data variability structure




```python
sim.config.model_parameters.alpha_species = Param(
    value=0.5, free=True, hyper=True,
    dims=('rabbit_species','experiment'),
    # take good care to specify hyperpriors correctly. 
    # Dimensions are broadcasted following the normal rules of 
    # numpy. The below means, in dimension one, we have two different
    # assumptions 1, and 3. Dimension one is the dimension of the rabbit species.
    # The specification loc=[1,3] would be understood as [[1,3]] and
    # be understood as the experiment dimension. Ideally, the dimensionality
    # is so low that you can be specific about the priors. I.e.:
    # scale = [[1,1,1],[3,3,3]]. This of course expects you know about
    # the dimensionality of the prior (i.e. the unique coordinates of the dimensions)
    prior="norm(loc=[[1],[3]],scale=0.1)" # type: ignore
)
# prey birth rate
# to be clear, this says each replicate has a slightly varying birth
# rate depending on the valley where it was observed. Seems legit.
sim.config.model_parameters.alpha = Param(
    value=0.5, free=True, hyper=False,
    dims=('id',),
    prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index])" # type: ignore
)

# re initialize the observation with
sim.define_observations_replicated_multi_experiment(n=120) # type: ignore
sim.coordinates["time"] = np.arange(12)

# This is a mistake 💥 as we will learn later on ('hierarchical_model_varying_y0.ipynb')
y0 = sim.parse_input("y0", drop_dims=["time"])
sim.model_parameters["y0"] = y0
```

Small teaser, we define the initial values from the noisy observations! Knowing the true starting values is essential, for correctly fitting the model. But let's go step by step. In the next part of the tutorial we'll take a look at varying initial values.

## Sample from the nested parameter distribution

To simply generate some parameter samples from a distribution, the ScipyBackend has been set up.


```python
inferer = ScipyBackend(simulation=sim)

theta = inferer.sample_distribution()

alpha_samples_cottontail = theta["alpha"][sim.observations["rabbit_species"] == "Cottontail"]
alpha_samples_jackrabbit = theta["alpha"][sim.observations["rabbit_species"] == "Jackrabbit"]

alpha_cottontail = np.mean(alpha_samples_cottontail)
alpha_jackrabbit = np.mean(alpha_samples_jackrabbit)

# test if the priors that were broadcasted to the replicates 
# match the hyperpriors
np.testing.assert_array_almost_equal(
    [alpha_cottontail, alpha_jackrabbit], [1, 3], decimal=1
)
```


```python
theta
```




    {'alpha_species': array([[1.03455842, 1.08216181, 1.03304371],
            [2.86968428, 3.09053559, 3.04463746]]),
     'alpha': array([0.98047254, 1.09645966, 1.07297153, 1.06544008, 1.03750305,
            1.09269376, 0.96110586, 1.01784098, 0.98586363, 1.0984052 ,
            1.03867608, 1.00474021, 0.95674713, 1.00828963, 1.03540112,
            1.00643501, 1.17748531, 1.14413297, 0.78887961, 0.85647801,
            2.81996594, 2.75105087, 2.93165267, 2.9327314 , 3.54658756,
            2.56767274, 2.76334393, 3.52006402, 3.06139996, 3.06641263,
            2.72590744, 2.43365562, 2.91814602, 2.90113902, 2.53822955,
            2.68016765, 2.84908431, 2.61098319, 2.84162201, 2.89721612,
            1.08601968, 1.02873671, 1.14836079, 1.18302819, 1.11744581,
            0.99714179, 1.16430687, 1.02923594, 1.18160866, 0.97217639,
            1.18578789, 1.0799928 , 0.95512394, 1.04872042, 1.08803242,
            1.11208858, 0.98092616, 0.96872299, 1.10397707, 1.0328126 ,
            3.16418325, 3.33441202, 2.62076345, 3.17016367, 3.49316814,
            2.9999383 , 2.84984027, 3.33198688, 3.16986518, 3.38019267,
            2.98566599, 2.66488946, 3.0567227 , 2.95577709, 3.33968599,
            3.15096164, 2.62546883, 2.74238532, 3.37610713, 3.30792435,
            0.96897658, 1.03293537, 1.08011429, 1.08258309, 1.12764763,
            1.05988251, 1.02329383, 1.00664669, 1.14807173, 0.82483169,
            1.01881885, 1.03645839, 0.89581139, 1.06800333, 0.96790765,
            1.12609284, 1.02015063, 1.10453543, 1.16695041, 1.07336917,
            2.78935314, 2.61679417, 3.62814045, 3.01094088, 2.84204919,
            3.08887684, 2.98691386, 3.31545894, 3.0549849 , 3.04882659,
            2.83466527, 3.19101371, 2.74558773, 3.2542791 , 3.54584177,
            2.61408264, 2.37918718, 3.23836868, 3.92816193, 2.75464712]),
     'beta': 0.017648710084435453}



Next up we use the samples to generate some trajectories and add Poisson noise on top of the data


```python
sim.solver = JaxSolver
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
sim.dispatch_constructor()
e = sim.dispatch(theta=theta)
e()

rng = np.random.default_rng(1)

# add noise. 
obs = e.results
obs.rabbits.values = rng.poisson(e.results.rabbits+1e-6)
obs.wolves.values = rng.poisson(e.results.wolves+1e-6)


sim.observations = obs
sim.config.data_structure.rabbits.observed = True
sim.config.data_structure.wolves.observed = True

# update settings
sim.config.case_study.scenario = "test_hierarchical_presimulated"
sim.config.create_directory("scenario", force=True)
sim.config.create_directory("results", force=True)
sim.config.model_parameters.beta.value = np.round(theta["beta"], 4)
sim.config.model_parameters.alpha.value = np.round(theta["alpha"], 2)
sim.config.model_parameters.alpha_species.value = np.round(theta["alpha_species"],2)

# store simulated results
sim.save_observations("simulated_data_hierarchical_species_year.nc", force=True)

# store settings
sim.config.save(force=True)
```

    /home/flo-schu/projects/pymob/pymob/simulation.py:546: UserWarning: The number of ODE states was not specified in the config file [simulation] > 'n_ode_states = <n>'. Extracted the return arguments ['dprey_dt', 'dpredator_dt'] from the source code. Setting 'n_ode_states=2.
      warnings.warn(


    Scenario directory exists at '/home/flo-schu/projects/pymob/case_studies/lotka_volterra_case_study/scenarios/test_hierarchical_presimulated'.
    Results directory exists at '/home/flo-schu/projects/pymob/case_studies/lotka_volterra_case_study/results/test_hierarchical_presimulated'.


## Defining an incorrect error distribution 💥

To see how to diagnose problems in a model, we deliberately specify an incorrect distribution that looks innocuous, but has two severe problems. One is obvious, the other one is a sneaky one.
Below is a conventionally used way to define error models. We center a lognormal error model around the means of the distribution. 


```python
sim.config.error_model.rabbits = "lognorm(scale=rabbits+EPS, s=0.1)"
sim.config.error_model.wolves = "lognorm(scale=wolves+EPS, s=0.1)"
sim.dispatch_constructor()
sim.set_inferer("numpyro")

```

    Jax 64 bit mode: False
    Absolute tolerance: 1e-07


First we simply try to fit the distribution, but run into a problem, **because the lognormal distribution does not support zero values**. We get a warning from the `check_log_likelihood` function from the numpyro backend. If we are unsure if our model is specified incorrectly, it is a good idea to use that function.


```python
try:
    sim.inferer.run()
    raise AssertionError(
        "This model should fail, because there are zero values in the"+
        "observations, hence the log-likelihood becomes nan, because there"+
        "is no support for the values"
    )
except RuntimeError:
    # check likelihoods of rabbits     
    loglik = sim.inferer.check_log_likelihood(theta)
    nan_liks = np.isnan(loglik[2]["rabbits_obs"]).sum()

    assert nan_liks > 0
    print(
        "Likelihood is not well defined, there are zeros in the "+
        "observations, while support excludes zeros. "
    )

```

         Trace Shapes:          
          Param Sites:          
         Sample Sites:          
    alpha_species dist   2   3 |
                 value   2   3 |
            alpha dist     120 |
                 value     120 |
             beta dist         |
                 value         |
      rabbits_obs dist 120  12 |
                 value 120  12 |
       wolves_obs dist 120  12 |
                 value 120  12 |
    Likelihood is not well defined, there are zeros in the observations, while support excludes zeros. 


    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:652: UserWarning: Site rabbits_obs: Out-of-support values provided to log prob method. The value argument should be within the support.
      mcmc.run(next(keys))
    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:652: UserWarning: Site wolves_obs: Out-of-support values provided to log prob method. The value argument should be within the support.
      mcmc.run(next(keys))
    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:934: UserWarning: Log-likelihoods ['rabbits_obs', 'wolves_obs'] contained NaN or inf values. The gradient based samplers will not be able to sample from this model. Make sure that all functions are numerically well behaved. Inspect the model with `jax.debug.print('{}',x)` https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html#exploring-debug-callback Or look at the functions step by step to find the position where jnp.grad(func)(x) evaluates to NaN
      warnings.warn(


This problem can be cured by simply incrementing the observations by a small value, but we can go deeper and investigate if the error model is actually a fitting description of the data. For this we generate some prior predictions to look at further problems in the model


```python
idata = sim.inferer.prior_predictions(n=100)

# first we test if numpyro predictions also match the specified priors
alpha_numpyro = idata.prior["alpha"].mean(("chain", "draw"))
alpha_numpyro_cottontail = np.mean(alpha_numpyro.values[sim.observations["rabbit_species"] == "Cottontail"])
alpha_numpyro_jackrabbit = np.mean(alpha_numpyro.values[sim.observations["rabbit_species"] == "Jackrabbit"])

# test if the priors that were broadcasted to the replicates 
# match the hyperpriors
np.testing.assert_array_almost_equal(
    [alpha_numpyro_cottontail, alpha_numpyro_jackrabbit], [1, 3], decimal=1
)
```

Next we plot the likelihoods of the different data variables. This helps to diagnose problems with multiple endpoints


```python
loglik = idata.log_likelihood.sum(("id", "time"))
fig = plot_pair(idata.prior, loglik, parameters=["alpha", "beta"])
fig.savefig(f"{sim.output_path}/bad_likelihood.png")
```


    
![png](hierarchical_model_files/hierarchical_model_18_0.png)
    


The problem is: due to the large scale differences in rabbits and wolves, the log-likelihoods end up very differently. This has to do with heteroskedasticity. The lognormal density becomes smaller at larger values to maintain the requirement that probability distributions integrate to 1. Here the wolves data variable will basically be meaningless, because the rabbits data variable is at such a high scale Scaling alone also does not resolve this problem, because due to the dynamic of the data variables, larger values will have a higher weight. This is not right. 🤯

## Defining a correct error distribution for the data by using a residual error model

As it turns out, the residuals of a poisson distributed variable can be transformed to a standard normal distributon by dividing with the square root of the random variables mean. 


```python
scaled_residuals = (sim.observations - e.results)/np.sqrt(e.results+1e-6)
scaled_residuals.wolves.plot()
```




    <matplotlib.collections.QuadMesh at 0x7f31eaaa5910>




    
![png](hierarchical_model_files/hierarchical_model_21_1.png)
    


The heatmap plot shows us that the residual are equally distributed through time and id. This looks perfect. This means there is no underlying dynamic governing the residuals. In pymob, we specify this relationship **by providing a transform of the observations of our error model**.


```python
sim.config.error_model.rabbits = "norm(loc=0, scale=1, obs=(obs-rabbits)/jnp.sqrt(rabbits+1e-6))"
sim.config.error_model.wolves = "norm(loc=0, scale=1, obs=(obs-wolves)/jnp.sqrt(wolves+1e-6))"

sim.dispatch_constructor()
sim.set_inferer("numpyro")
```

    Jax 64 bit mode: False
    Absolute tolerance: 1e-07



```python
idata = sim.inferer.prior_predictions(n=100)

# no nan problems any longer in the likelihood
loglik = sim.inferer.check_log_likelihood(theta)
nan_liks_rabbits = np.isnan(loglik[2]["rabbits_obs"]).sum()
nan_liks_wolves = np.isnan(loglik[2]["wolves_obs"]).sum()
np.testing.assert_array_equal([nan_liks_wolves, nan_liks_rabbits], [0,0])

# plot likelihoods
loglik = idata.log_likelihood.mean(("id", "time"))
fig = plot_pair(idata.prior, loglik, parameters=["alpha", "beta"])
fig.savefig(f"{sim.output_path}/good_likelihood.png")
```

    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:1033: UserWarning: Cannot make predictions of observations from normalized observations (residuals). Please provide an inverse observation transform: e.g. `sim.config.error_model['rabbits'].obs_inv = ...`.residuals are denoted as 'res'. See Lotka-volterra case study for an example. 
      warnings.warn(
    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:1033: UserWarning: Cannot make predictions of observations from normalized observations (residuals). Please provide an inverse observation transform: e.g. `sim.config.error_model['wolves'].obs_inv = ...`.residuals are denoted as 'res'. See Lotka-volterra case study for an example. 
      warnings.warn(



    
![png](hierarchical_model_files/hierarchical_model_24_1.png)
    


Next we look at the problem from a slightly different angle. By splitting the likelihood between different ids (in case of a hierarchical model this is possible, we can look at problematic samples.)


```python
from scipy.stats import norm

# the 2nd visualization is actually not so helpful, because it rather focuses on
# the individual replicates and not so much on the dynamics of the parameters

idata = sim.inferer.prior_predictions(n=100, seed=132)

resid = (idata.prior_predictive.wolves - idata.observed_data.wolves)/np.sqrt(idata.prior_predictive.wolves)
loglik = norm(0,1).logpdf(resid)

idata.log_likelihood["wolves_recompute"] = (("chain", "draw","id", "time"), loglik)

loglik = idata.log_likelihood.sum(("time"))
# prior = idata.prior.rename({"alpha_dim_0":"id"})
fig = plot_pair(idata.prior, loglik, parameters=["alpha", "beta"])
fig.savefig(f"{sim.output_path}/better_likelihood_questionmark.png")
```

    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:1033: UserWarning: Cannot make predictions of observations from normalized observations (residuals). Please provide an inverse observation transform: e.g. `sim.config.error_model['rabbits'].obs_inv = ...`.residuals are denoted as 'res'. See Lotka-volterra case study for an example. 
      warnings.warn(
    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:1033: UserWarning: Cannot make predictions of observations from normalized observations (residuals). Please provide an inverse observation transform: e.g. `sim.config.error_model['wolves'].obs_inv = ...`.residuals are denoted as 'res'. See Lotka-volterra case study for an example. 
      warnings.warn(
    /home/flo-schu/miniconda3/envs/pymob/lib/python3.11/site-packages/xarray/core/computation.py:821: RuntimeWarning: invalid value encountered in sqrt
      result_data = func(*input_data)



    
![png](hierarchical_model_files/hierarchical_model_26_1.png)
    


Overall we conclude that it is way better to use residuals for the error modelling, because if the residuals are described correctly, this results in an equally distributed likelihood of the errorrs.

In addition, the reparameterization of the error distribution has seemed to help the NUTS sampler.


```python
# fitting with SVI seems to work okay
sim.config.inference_numpyro.svi_iterations = 2_000
sim.config.inference_numpyro.svi_learning_rate = 0.005
sim.config.inference_numpyro.gaussian_base_distribution = True
sim.config.jaxsolver.max_steps = 1e5
sim.config.jaxsolver.throw_exception = False
sim.config.inference_numpyro.init_strategy = "init_to_median"
sim.dispatch_constructor()
sim.set_inferer("numpyro")

sample_nuts = False
if sample_nuts:
    sim.config.inference_numpyro.kernel = "nuts"
    sim.inferer.run()
    sim.inferer.store_results() # type: ignore
else:
    sim.inferer.load_results()

idata_nuts = sim.inferer.idata.copy()
az.summary(sim.inferer.idata.posterior)
```

    /home/flo-schu/miniconda3/envs/pymob/lib/python3.11/site-packages/pydantic/main.py:308: UserWarning: Pydantic serializer warnings:
      Expected `int` but got `float` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(


    Jax 64 bit mode: False
    Absolute tolerance: 1e-07


    arviz - WARNING - Shape validation failed: input_shape: (1, 2000), minimum_shape: (chains=2, draws=4)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha[0]</th>
      <td>0.963</td>
      <td>0.018</td>
      <td>0.928</td>
      <td>0.995</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3182.0</td>
      <td>1572.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>alpha[1]</th>
      <td>1.085</td>
      <td>0.015</td>
      <td>1.055</td>
      <td>1.112</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3676.0</td>
      <td>1873.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>alpha[2]</th>
      <td>1.026</td>
      <td>0.019</td>
      <td>0.994</td>
      <td>1.065</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2785.0</td>
      <td>1505.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>alpha[3]</th>
      <td>1.051</td>
      <td>0.019</td>
      <td>1.017</td>
      <td>1.086</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3544.0</td>
      <td>1823.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>alpha[4]</th>
      <td>1.022</td>
      <td>0.014</td>
      <td>0.996</td>
      <td>1.049</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3434.0</td>
      <td>1287.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>wolves_res[119, 7]</th>
      <td>0.159</td>
      <td>0.135</td>
      <td>-0.088</td>
      <td>0.420</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3380.0</td>
      <td>1368.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>wolves_res[119, 8]</th>
      <td>0.911</td>
      <td>0.121</td>
      <td>0.690</td>
      <td>1.144</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3380.0</td>
      <td>1368.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>wolves_res[119, 9]</th>
      <td>-0.200</td>
      <td>0.099</td>
      <td>-0.380</td>
      <td>-0.010</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3380.0</td>
      <td>1368.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>wolves_res[119, 10]</th>
      <td>0.179</td>
      <td>0.087</td>
      <td>0.021</td>
      <td>0.347</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3380.0</td>
      <td>1368.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>wolves_res[119, 11]</th>
      <td>0.911</td>
      <td>0.079</td>
      <td>0.767</td>
      <td>1.063</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3379.0</td>
      <td>1368.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>6014 rows × 9 columns</p>
</div>




```python

az.plot_trace(idata_nuts, var_names=("alpha_species", "beta", "alpha"))
```




    array([[<Axes: title={'center': 'alpha_species'}>,
            <Axes: title={'center': 'alpha_species'}>],
           [<Axes: title={'center': 'beta'}>,
            <Axes: title={'center': 'beta'}>],
           [<Axes: title={'center': 'alpha'}>,
            <Axes: title={'center': 'alpha'}>]], dtype=object)




    
![png](hierarchical_model_files/hierarchical_model_29_1.png)
    


The parameters are perfectly recovered. We have a true beta of 0.1765 and the fitted beta is 0.1755, where the distribution contains the true parameter, although the mode is a bit off. I'm curious if the residual error distribution was too wide or too narrow would have made the posterior beta distribution wider. Also in a second iteration, the priors for estimating should be made less informative, to see if the inference still works. But overall this has been a success. We have no divergences, perfect r_hat and high effective sampling size. So things look good


```python
theta["beta"]
```




    0.017648710084435453




```python
posterior = idata_nuts.posterior[["alpha", "beta"]].rename({"alpha_dim_0": "id"})
loglik = idata_nuts.log_likelihood.mean(("time"))
fig = plot_pair(posterior, loglik, parameters=["alpha", "beta"])
fig.savefig(f"{sim.output_path}/posterior.png")
```


    
![png](hierarchical_model_files/hierarchical_model_32_0.png)
    


## Inspect fitted results from MCMC


```python
# fitting with SVI seems to work okay
sim.config.inference_numpyro.kernel = "svi"
sim.config.inference_numpyro.svi_iterations = 2_000
sim.config.inference_numpyro.svi_learning_rate = 0.005
sim.config.inference_numpyro.gaussian_base_distribution = True
sim.config.jaxsolver.max_steps = 1e5
sim.config.jaxsolver.throw_exception = False
sim.config.inference_numpyro.init_strategy = "init_to_median"
sim.dispatch_constructor()
sim.set_inferer("numpyro")
sim.inferer.run()
idata_svi = sim.inferer.idata.copy()
```

    /home/flo-schu/miniconda3/envs/pymob/lib/python3.11/site-packages/pydantic/main.py:308: UserWarning: Pydantic serializer warnings:
      Expected `int` but got `float` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(


    Jax 64 bit mode: False
    Absolute tolerance: 1e-07
                     Trace Shapes:          
                      Param Sites:          
                     Sample Sites:          
    alpha_species_normal_base dist   2   3 |
                             value   2   3 |
            alpha_normal_base dist     120 |
                             value     120 |
             beta_normal_base dist         |
                             value         |
                  rabbits_obs dist 120  12 |
                             value 120  12 |
                   wolves_obs dist 120  12 |
                             value 120  12 |


    100%|██████████| 2000/2000 [00:24<00:00, 82.05it/s, init loss: 12252.5215, avg. loss [1901-2000]: 4062.4299] 
    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:1033: UserWarning: Cannot make predictions of observations from normalized observations (residuals). Please provide an inverse observation transform: e.g. `sim.config.error_model['rabbits'].obs_inv = ...`.residuals are denoted as 'res'. See Lotka-volterra case study for an example. 
      warnings.warn(
    /home/flo-schu/projects/pymob/pymob/inference/numpyro_backend.py:1033: UserWarning: Cannot make predictions of observations from normalized observations (residuals). Please provide an inverse observation transform: e.g. `sim.config.error_model['wolves'].obs_inv = ...`.residuals are denoted as 'res'. See Lotka-volterra case study for an example. 
      warnings.warn(
    arviz - WARNING - Shape validation failed: input_shape: (1, 2000), minimum_shape: (chains=2, draws=4)


                                      mean     sd  hdi_3%  hdi_97%  mcse_mean  \
    alpha[0]                         0.969  0.017   0.937    1.001        0.0   
    alpha[1]                         1.083  0.016   1.054    1.114        0.0   
    alpha[2]                         1.025  0.019   0.990    1.061        0.0   
    alpha[3]                         1.048  0.018   1.015    1.081        0.0   
    alpha[4]                         1.021  0.015   0.993    1.048        0.0   
    ...                                ...    ...     ...      ...        ...   
    alpha_species[Cottontail, 2012]  1.028  0.011   1.008    1.048        0.0   
    alpha_species[Jackrabbit, 2010]  2.908  0.018   2.876    2.943        0.0   
    alpha_species[Jackrabbit, 2011]  3.047  0.017   3.015    3.081        0.0   
    alpha_species[Jackrabbit, 2012]  3.054  0.014   3.028    3.081        0.0   
    beta                             0.018  0.000   0.017    0.018        0.0   
    
                                     mcse_sd  ess_bulk  ess_tail  r_hat  
    alpha[0]                             0.0    1972.0    2046.0    NaN  
    alpha[1]                             0.0    1947.0    1450.0    NaN  
    alpha[2]                             0.0    2097.0    2004.0    NaN  
    alpha[3]                             0.0    1710.0    1655.0    NaN  
    alpha[4]                             0.0    2039.0    1962.0    NaN  
    ...                                  ...       ...       ...    ...  
    alpha_species[Cottontail, 2012]      0.0    1901.0    1717.0    NaN  
    alpha_species[Jackrabbit, 2010]      0.0    1864.0    1915.0    NaN  
    alpha_species[Jackrabbit, 2011]      0.0    1940.0    1961.0    NaN  
    alpha_species[Jackrabbit, 2012]      0.0    2105.0    1931.0    NaN  
    beta                                 0.0    2032.0    1961.0    NaN  
    
    [127 rows x 9 columns]



    
![png](hierarchical_model_files/hierarchical_model_34_4.png)
    



```python
posteriors = xr.combine_by_coords([
    idata_svi.posterior.expand_dims("algorithm").assign_coords({"algorithm": ["svi"]}),
    idata_nuts.posterior.expand_dims("algorithm").assign_coords({"algorithm": ["nuts"]}),
    ], combine_attrs="drop"
)
posteriors

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
Dimensions:                          (chain: 1, draw: 2000, alpha_dim_0: 120,
                                      alpha_normal_base_dim_0: 120,
                                      alpha_species_dim_0: 2,
                                      alpha_species_dim_1: 3,
                                      alpha_species_normal_base_dim_0: 2,
                                      alpha_species_normal_base_dim_1: 3,
                                      id: 120, time: 12,
                                      rabbits_res_dim_0: 120,
                                      rabbits_res_dim_1: 12,
                                      wolves_res_dim_0: 120,
                                      wolves_res_dim_1: 12, algorithm: 2,
                                      rabbit_species: 2, experiment: 3)
Coordinates: (12/19)
  * chain                            (chain) int64 0
  * draw                             (draw) int64 0 1 2 3 ... 1997 1998 1999
  * alpha_dim_0                      (alpha_dim_0) int64 0 1 2 3 ... 117 118 119
  * alpha_normal_base_dim_0          (alpha_normal_base_dim_0) int64 0 1 ... 119
  * alpha_species_dim_0              (alpha_species_dim_0) int64 0 1
  * alpha_species_dim_1              (alpha_species_dim_1) int64 0 1 2
    ...                               ...
  * wolves_res_dim_1                 (wolves_res_dim_1) int64 0 1 2 ... 9 10 11
  * algorithm                        (algorithm) &lt;U4 &#x27;nuts&#x27; &#x27;svi&#x27;
  * rabbit_species                   (rabbit_species) &lt;U10 &#x27;Cottontail&#x27; &#x27;Jack...
  * experiment                       (experiment) &lt;U4 &#x27;2010&#x27; &#x27;2011&#x27; &#x27;2012&#x27;
    rabbit_species_index             (id) int64 0 0 0 0 0 0 0 ... 1 1 1 1 1 1 1
    experiment_index                 (id) int64 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2
Data variables:
    alpha                            (algorithm, chain, draw, alpha_dim_0, id) float32 ...
    alpha_normal_base                (algorithm, chain, draw, alpha_normal_base_dim_0) float32 ...
    alpha_species                    (algorithm, chain, draw, alpha_species_dim_0, alpha_species_dim_1, rabbit_species, experiment) float32 ...
    alpha_species_normal_base        (algorithm, chain, draw, alpha_species_normal_base_dim_0, alpha_species_normal_base_dim_1) float32 ...
    beta                             (algorithm, chain, draw) float32 0.01757...
    beta_normal_base                 (algorithm, chain, draw) float32 -1.297 ...
    rabbits                          (algorithm, chain, draw, id, time) float32 ...
    rabbits_res                      (algorithm, chain, draw, rabbits_res_dim_0, rabbits_res_dim_1) float32 ...
    wolves                           (algorithm, chain, draw, id, time) float32 ...
    wolves_res                       (algorithm, chain, draw, wolves_res_dim_0, wolves_res_dim_1) float32 ...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-46e674a4-98d4-4769-a36f-a8e793373646' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-46e674a4-98d4-4769-a36f-a8e793373646' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 2000</li><li><span class='xr-has-index'>alpha_dim_0</span>: 120</li><li><span class='xr-has-index'>alpha_normal_base_dim_0</span>: 120</li><li><span class='xr-has-index'>alpha_species_dim_0</span>: 2</li><li><span class='xr-has-index'>alpha_species_dim_1</span>: 3</li><li><span class='xr-has-index'>alpha_species_normal_base_dim_0</span>: 2</li><li><span class='xr-has-index'>alpha_species_normal_base_dim_1</span>: 3</li><li><span class='xr-has-index'>id</span>: 120</li><li><span class='xr-has-index'>time</span>: 12</li><li><span class='xr-has-index'>rabbits_res_dim_0</span>: 120</li><li><span class='xr-has-index'>rabbits_res_dim_1</span>: 12</li><li><span class='xr-has-index'>wolves_res_dim_0</span>: 120</li><li><span class='xr-has-index'>wolves_res_dim_1</span>: 12</li><li><span class='xr-has-index'>algorithm</span>: 2</li><li><span class='xr-has-index'>rabbit_species</span>: 2</li><li><span class='xr-has-index'>experiment</span>: 3</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-51b9aa63-9e9e-474b-a1a3-0a194b21ab51' class='xr-section-summary-in' type='checkbox'  checked><label for='section-51b9aa63-9e9e-474b-a1a3-0a194b21ab51' class='xr-section-summary' >Coordinates: <span>(19)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-fb507ac2-70b1-4af8-b5c1-7d6d43f2f939' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fb507ac2-70b1-4af8-b5c1-7d6d43f2f939' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f58c76ab-bae6-427e-ae07-1c00c8ec69fc' class='xr-var-data-in' type='checkbox'><label for='data-f58c76ab-bae6-427e-ae07-1c00c8ec69fc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1996 1997 1998 1999</div><input id='attrs-82462e72-1894-487b-bd09-285882bdcdb9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-82462e72-1894-487b-bd09-285882bdcdb9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-046ad18d-f7c5-4c82-b0bd-654ec3aa6d65' class='xr-var-data-in' type='checkbox'><label for='data-046ad18d-f7c5-4c82-b0bd-654ec3aa6d65' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1997, 1998, 1999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>alpha_dim_0</span></div><div class='xr-var-dims'>(alpha_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 115 116 117 118 119</div><input id='attrs-2cd46bac-168d-439b-ab3d-b395f7f95d5b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2cd46bac-168d-439b-ab3d-b395f7f95d5b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5bb78909-4c25-4e76-9f0c-3f7da3ddf4d0' class='xr-var-data-in' type='checkbox'><label for='data-5bb78909-4c25-4e76-9f0c-3f7da3ddf4d0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>alpha_normal_base_dim_0</span></div><div class='xr-var-dims'>(alpha_normal_base_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 115 116 117 118 119</div><input id='attrs-095d778e-0d4c-4946-9e5d-ef728695d511' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-095d778e-0d4c-4946-9e5d-ef728695d511' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-39ce6e3f-14ca-4fab-a953-6eb4ffe748d0' class='xr-var-data-in' type='checkbox'><label for='data-39ce6e3f-14ca-4fab-a953-6eb4ffe748d0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>alpha_species_dim_0</span></div><div class='xr-var-dims'>(alpha_species_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-0b1eed1e-5c7f-4312-8e30-4e4cf503c081' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0b1eed1e-5c7f-4312-8e30-4e4cf503c081' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-742ccfdb-c461-4093-a853-926a8e60dac3' class='xr-var-data-in' type='checkbox'><label for='data-742ccfdb-c461-4093-a853-926a8e60dac3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>alpha_species_dim_1</span></div><div class='xr-var-dims'>(alpha_species_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2</div><input id='attrs-2e3603ec-c384-4c41-a2e1-aa7acb6e6b89' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2e3603ec-c384-4c41-a2e1-aa7acb6e6b89' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-febd16ee-b40b-4ded-ad48-4b87d680cdaf' class='xr-var-data-in' type='checkbox'><label for='data-febd16ee-b40b-4ded-ad48-4b87d680cdaf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>alpha_species_normal_base_dim_0</span></div><div class='xr-var-dims'>(alpha_species_normal_base_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1</div><input id='attrs-33cca717-bb84-4b1e-ac93-3b872cd00511' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-33cca717-bb84-4b1e-ac93-3b872cd00511' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9b65fb81-10a7-4295-b307-2e19fba5f5a4' class='xr-var-data-in' type='checkbox'><label for='data-9b65fb81-10a7-4295-b307-2e19fba5f5a4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>alpha_species_normal_base_dim_1</span></div><div class='xr-var-dims'>(alpha_species_normal_base_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2</div><input id='attrs-3a17c308-c1a1-4649-a419-7f96364fb8d0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3a17c308-c1a1-4649-a419-7f96364fb8d0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-91227c9a-d225-43c1-9dd1-3bf7f0808cf4' class='xr-var-data-in' type='checkbox'><label for='data-91227c9a-d225-43c1-9dd1-3bf7f0808cf4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>id</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 115 116 117 118 119</div><input id='attrs-115071c6-6293-40dd-902b-04030af1d95f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-115071c6-6293-40dd-902b-04030af1d95f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-aafddb33-9674-431a-8cba-8004f8bfcbc9' class='xr-var-data-in' type='checkbox'><label for='data-aafddb33-9674-431a-8cba-8004f8bfcbc9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9 10 11</div><input id='attrs-7c87dcf1-4860-41d7-b871-3da932849f95' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7c87dcf1-4860-41d7-b871-3da932849f95' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dc85fdfe-1c41-4b61-9a5d-dd85bad77d6a' class='xr-var-data-in' type='checkbox'><label for='data-dc85fdfe-1c41-4b61-9a5d-dd85bad77d6a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>rabbits_res_dim_0</span></div><div class='xr-var-dims'>(rabbits_res_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 115 116 117 118 119</div><input id='attrs-8d3a6a9e-0400-4a1c-a811-6828cd0e03d3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8d3a6a9e-0400-4a1c-a811-6828cd0e03d3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-05f52589-dcc0-4197-970e-2d34584faa54' class='xr-var-data-in' type='checkbox'><label for='data-05f52589-dcc0-4197-970e-2d34584faa54' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>rabbits_res_dim_1</span></div><div class='xr-var-dims'>(rabbits_res_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9 10 11</div><input id='attrs-be55ae75-b25f-4686-b838-e580c1eddc65' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-be55ae75-b25f-4686-b838-e580c1eddc65' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a12bdf53-91da-4643-9c2c-ced59b0c565b' class='xr-var-data-in' type='checkbox'><label for='data-a12bdf53-91da-4643-9c2c-ced59b0c565b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>wolves_res_dim_0</span></div><div class='xr-var-dims'>(wolves_res_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 115 116 117 118 119</div><input id='attrs-878867ff-1789-415f-a959-42721e1a1615' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-878867ff-1789-415f-a959-42721e1a1615' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ddc41bdf-f90b-458d-a2d5-0f9712e8a06c' class='xr-var-data-in' type='checkbox'><label for='data-ddc41bdf-f90b-458d-a2d5-0f9712e8a06c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>wolves_res_dim_1</span></div><div class='xr-var-dims'>(wolves_res_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9 10 11</div><input id='attrs-b56e88a4-2ae5-4e07-a9be-1142a9428432' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b56e88a4-2ae5-4e07-a9be-1142a9428432' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-38bf85db-183e-4c2f-b658-104a1a5c023c' class='xr-var-data-in' type='checkbox'><label for='data-38bf85db-183e-4c2f-b658-104a1a5c023c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>algorithm</span></div><div class='xr-var-dims'>(algorithm)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;nuts&#x27; &#x27;svi&#x27;</div><input id='attrs-074d7c31-aee5-4e0b-a589-911fd8127be0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-074d7c31-aee5-4e0b-a589-911fd8127be0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8c7fa995-58e0-4a76-89a1-d1bca1f2e4fb' class='xr-var-data-in' type='checkbox'><label for='data-8c7fa995-58e0-4a76-89a1-d1bca1f2e4fb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;nuts&#x27;, &#x27;svi&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>rabbit_species</span></div><div class='xr-var-dims'>(rabbit_species)</div><div class='xr-var-dtype'>&lt;U10</div><div class='xr-var-preview xr-preview'>&#x27;Cottontail&#x27; &#x27;Jackrabbit&#x27;</div><input id='attrs-3776e67d-e2fb-4e4a-a06b-898337249a4b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3776e67d-e2fb-4e4a-a06b-898337249a4b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-22a5f625-8d0a-4500-bcbf-2a83ee031f1d' class='xr-var-data-in' type='checkbox'><label for='data-22a5f625-8d0a-4500-bcbf-2a83ee031f1d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Cottontail&#x27;, &#x27;Jackrabbit&#x27;], dtype=&#x27;&lt;U10&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>experiment</span></div><div class='xr-var-dims'>(experiment)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;2010&#x27; &#x27;2011&#x27; &#x27;2012&#x27;</div><input id='attrs-c0801850-9155-49c5-9bec-252a74dca41d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c0801850-9155-49c5-9bec-252a74dca41d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7ef8b988-6088-4d18-a51d-d29cd3805e0a' class='xr-var-data-in' type='checkbox'><label for='data-7ef8b988-6088-4d18-a51d-d29cd3805e0a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2010&#x27;, &#x27;2011&#x27;, &#x27;2012&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>rabbit_species_index</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 0 ... 1 1 1 1 1 1 1 1</div><input id='attrs-4972bb70-a49b-4f9c-9f19-af4e22f199bc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4972bb70-a49b-4f9c-9f19-af4e22f199bc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-53aa9ae5-e509-48de-90c8-2fd0f1a7da4b' class='xr-var-data-in' type='checkbox'><label for='data-53aa9ae5-e509-48de-90c8-2fd0f1a7da4b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>experiment_index</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2</div><input id='attrs-87451d7a-f5c6-4546-912e-e22eaaff4d12' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-87451d7a-f5c6-4546-912e-e22eaaff4d12' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-db1cb124-acd0-4715-a9f9-c572c63233dc' class='xr-var-data-in' type='checkbox'><label for='data-db1cb124-acd0-4715-a9f9-c572c63233dc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-9f6a4ca3-5c51-43f4-8c88-171dbc187245' class='xr-section-summary-in' type='checkbox'  checked><label for='section-9f6a4ca3-5c51-43f4-8c88-171dbc187245' class='xr-section-summary' >Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>alpha</span></div><div class='xr-var-dims'>(algorithm, chain, draw, alpha_dim_0, id)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.9641 0.9641 0.9641 ... 3.895 2.73</div><input id='attrs-07ca2ee0-34bb-45e9-848e-b89362fa17c2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-07ca2ee0-34bb-45e9-848e-b89362fa17c2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1c87387c-5e09-4cab-aee7-4b3425e041c3' class='xr-var-data-in' type='checkbox'><label for='data-1c87387c-5e09-4cab-aee7-4b3425e041c3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[[0.96406555, 0.96406555, 0.96406555, ..., 0.96406555,
           0.96406555, 0.96406555],
          [1.0764744 , 1.0764744 , 1.0764744 , ..., 1.0764744 ,
           1.0764744 , 1.0764744 ],
          [1.0106924 , 1.0106924 , 1.0106924 , ..., 1.0106924 ,
           1.0106924 , 1.0106924 ],
          ...,
          [3.2320938 , 3.2320938 , 3.2320938 , ..., 3.2320938 ,
           3.2320938 , 3.2320938 ],
          [3.912693  , 3.912693  , 3.912693  , ..., 3.912693  ,
           3.912693  , 3.912693  ],
          [2.7933557 , 2.7933557 , 2.7933557 , ..., 2.7933557 ,
           2.7933557 , 2.7933557 ]],

         [[0.9514953 , 0.9514953 , 0.9514953 , ..., 0.9514953 ,
           0.9514953 , 0.9514953 ],
          [1.079344  , 1.079344  , 1.079344  , ..., 1.079344  ,
           1.079344  , 1.079344  ],
          [1.0155098 , 1.0155098 , 1.0155098 , ..., 1.0155098 ,
           1.0155098 , 1.0155098 ],
...
          [0.9454221 , 1.0487151 , 0.98559004, ..., 3.3036103 ,
           3.9354777 , 2.7079892 ],
          [0.9454221 , 1.0487151 , 0.98559004, ..., 3.3036103 ,
           3.9354777 , 2.7079892 ],
          [0.9454221 , 1.0487151 , 0.98559004, ..., 3.3036103 ,
           3.9354777 , 2.7079892 ]],

         [[0.9840798 , 1.0894021 , 1.0285487 , ..., 3.3044252 ,
           3.8949094 , 2.7301116 ],
          [0.9840798 , 1.0894021 , 1.0285487 , ..., 3.3044252 ,
           3.8949094 , 2.7301116 ],
          [0.9840798 , 1.0894021 , 1.0285487 , ..., 3.3044252 ,
           3.8949094 , 2.7301116 ],
          ...,
          [0.9840798 , 1.0894021 , 1.0285487 , ..., 3.3044252 ,
           3.8949094 , 2.7301116 ],
          [0.9840798 , 1.0894021 , 1.0285487 , ..., 3.3044252 ,
           3.8949094 , 2.7301116 ],
          [0.9840798 , 1.0894021 , 1.0285487 , ..., 3.3044252 ,
           3.8949094 , 2.7301116 ]]]]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>alpha_normal_base</span></div><div class='xr-var-dims'>(algorithm, chain, draw, alpha_normal_base_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-0.2296 0.8733 0.2427 ... nan nan</div><input id='attrs-bd8cc948-7ef6-4f0d-9dbc-4d415e701746' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bd8cc948-7ef6-4f0d-9dbc-4d415e701746' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-979dc898-465a-4aea-8754-a8c8827e47ff' class='xr-var-data-in' type='checkbox'><label for='data-979dc898-465a-4aea-8754-a8c8827e47ff' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[-0.22958985,  0.8732834 ,  0.2427266 , ...,  0.69379026,
           2.6047468 , -0.7650752 ],
         [-0.4434993 ,  0.8172411 ,  0.20761424, ...,  0.61955965,
           2.4466817 , -0.7764242 ],
         [-0.503253  ,  0.3190473 , -0.37744328, ...,  0.76074845,
           2.4291523 , -1.055161  ],
         ...,
         [-0.56927806,  0.94393384,  0.03041685, ...,  0.89878684,
           2.5929503 , -1.030673  ],
         [-0.42581767,  0.93156713,  0.37949994, ...,  0.904224  ,
           2.6418357 , -0.57077545],
         [-0.44493324,  0.48618895, -0.07270492, ...,  0.7358019 ,
           2.519345  , -1.3219705 ]]],


       [[[        nan,         nan,         nan, ...,         nan,
                  nan,         nan],
         [        nan,         nan,         nan, ...,         nan,
                  nan,         nan],
         [        nan,         nan,         nan, ...,         nan,
                  nan,         nan],
         ...,
         [        nan,         nan,         nan, ...,         nan,
                  nan,         nan],
         [        nan,         nan,         nan, ...,         nan,
                  nan,         nan],
         [        nan,         nan,         nan, ...,         nan,
                  nan,         nan]]]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>alpha_species</span></div><div class='xr-var-dims'>(algorithm, chain, draw, alpha_species_dim_0, alpha_species_dim_1, rabbit_species, experiment)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.9865 0.9865 ... 3.038 3.061</div><input id='attrs-cb9c7f0a-d506-45de-a896-89b52eb1c4d0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cb9c7f0a-d506-45de-a896-89b52eb1c4d0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c4588e4a-27d0-4dc9-95b3-60031b364ce0' class='xr-var-data-in' type='checkbox'><label for='data-c4588e4a-27d0-4dc9-95b3-60031b364ce0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[[[[0.98645556, 0.98645556, 0.98645556],
            [0.98645556, 0.98645556, 0.98645556]],

           [[1.0280201 , 1.0280201 , 1.0280201 ],
            [1.0280201 , 1.0280201 , 1.0280201 ]],

           [[1.0568117 , 1.0568117 , 1.0568117 ],
            [1.0568117 , 1.0568117 , 1.0568117 ]]],


          [[[2.8725977 , 2.8725977 , 2.8725977 ],
            [2.8725977 , 2.8725977 , 2.8725977 ]],

           [[3.0025485 , 3.0025485 , 3.0025485 ],
            [3.0025485 , 3.0025485 , 3.0025485 ]],

           [[3.0154564 , 3.0154564 , 3.0154564 ],
            [3.0154564 , 3.0154564 , 3.0154564 ]]]],


...


         [[[[1.015507  , 1.0411904 , 1.0297841 ],
            [2.9158635 , 3.0380332 , 3.0605016 ]],

           [[1.015507  , 1.0411904 , 1.0297841 ],
            [2.9158635 , 3.0380332 , 3.0605016 ]],

           [[1.015507  , 1.0411904 , 1.0297841 ],
            [2.9158635 , 3.0380332 , 3.0605016 ]]],


          [[[1.015507  , 1.0411904 , 1.0297841 ],
            [2.9158635 , 3.0380332 , 3.0605016 ]],

           [[1.015507  , 1.0411904 , 1.0297841 ],
            [2.9158635 , 3.0380332 , 3.0605016 ]],

           [[1.015507  , 1.0411904 , 1.0297841 ],
            [2.9158635 , 3.0380332 , 3.0605016 ]]]]]]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>alpha_species_normal_base</span></div><div class='xr-var-dims'>(algorithm, chain, draw, alpha_species_normal_base_dim_0, alpha_species_normal_base_dim_1)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-0.1354 0.2802 0.5681 ... nan nan</div><input id='attrs-fa9c71e3-3920-4e99-8228-79e940538520' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fa9c71e3-3920-4e99-8228-79e940538520' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-02965eea-39e5-453c-a7ba-fdc57d5b02fb' class='xr-var-data-in' type='checkbox'><label for='data-02965eea-39e5-453c-a7ba-fdc57d5b02fb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[[-0.13544449,  0.28020167,  0.56811714],
          [-1.2740232 ,  0.02548388,  0.15456441]],

         [[-0.05356184,  0.27246857,  0.46996218],
          [-1.5207386 ,  0.4167514 ,  0.63923526]],

         [[ 0.51100063,  0.8363559 ,  0.3179709 ],
          [-0.0391493 ,  1.2025334 ,  0.49876842]],

         ...,

         [[-0.18730223,  0.6476182 , -0.17390211],
          [-1.4416806 ,  0.3514304 ,  0.02586368]],

         [[ 0.00571556,  0.35003537,  0.8382446 ],
          [-1.7634724 ,  0.9123134 ,  0.01625257]],

         [[ 0.10459955,  0.59028596,  0.03624374],
          [-1.2069894 , -0.09897961,  0.5881367 ]]]],

...

       [[[[        nan,         nan,         nan],
          [        nan,         nan,         nan]],

         [[        nan,         nan,         nan],
          [        nan,         nan,         nan]],

         [[        nan,         nan,         nan],
          [        nan,         nan,         nan]],

         ...,

         [[        nan,         nan,         nan],
          [        nan,         nan,         nan]],

         [[        nan,         nan,         nan],
          [        nan,         nan,         nan]],

         [[        nan,         nan,         nan],
          [        nan,         nan,         nan]]]]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta</span></div><div class='xr-var-dims'>(algorithm, chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.01757 0.01756 ... 0.01757 0.01749</div><input id='attrs-eedb9933-24f9-4eab-8f86-c8ee80da9782' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eedb9933-24f9-4eab-8f86-c8ee80da9782' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a93f1fde-d6c8-4263-8a1d-607a3e6aaa98' class='xr-var-data-in' type='checkbox'><label for='data-a93f1fde-d6c8-4263-8a1d-607a3e6aaa98' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[0.01756706, 0.01755571, 0.01756362, ..., 0.01757979,
         0.01753608, 0.01753484]],

       [[0.01757514, 0.01757063, 0.01749719, ..., 0.01754425,
         0.01756923, 0.01749126]]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>beta_normal_base</span></div><div class='xr-var-dims'>(algorithm, chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-1.297 -1.304 -1.299 ... nan nan</div><input id='attrs-f0c16ead-e083-428f-b713-683959366efd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f0c16ead-e083-428f-b713-683959366efd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2a0f0eaf-904a-4756-ad6a-9decb1fef2f4' class='xr-var-data-in' type='checkbox'><label for='data-2a0f0eaf-904a-4756-ad6a-9decb1fef2f4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[-1.2970665, -1.3035282, -1.2990214, ..., -1.2898183,
         -1.3147156, -1.3154198]],

       [[       nan,        nan,        nan, ...,        nan,
                nan,        nan]]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>rabbits</span></div><div class='xr-var-dims'>(algorithm, chain, draw, id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>56.0 122.9 197.0 ... nan nan nan</div><input id='attrs-22f4a812-30a3-415f-a7b4-398089502d53' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-22f4a812-30a3-415f-a7b4-398089502d53' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-000d9cb0-7a1d-45b4-b892-69f8ee62e1a8' class='xr-var-data-in' type='checkbox'><label for='data-000d9cb0-7a1d-45b4-b892-69f8ee62e1a8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[[5.60000000e+01, 1.22897232e+02, 1.97035858e+02, ...,
           3.25670272e-01, 4.67074424e-01, 7.81616986e-01],
          [4.40000000e+01, 1.09396851e+02, 2.12893051e+02, ...,
           1.83141127e-01, 2.73173034e-01, 4.84889328e-01],
          [6.70000000e+01, 1.41258636e+02, 1.67516739e+02, ...,
           5.84459305e-01, 9.29204464e-01, 1.69576526e+00],
          ...,
          [5.80000000e+01, 6.82561157e+02, 1.58774987e-01, ...,
           1.16978129e-06, 8.49270236e-06, 8.56619736e-05],
          [5.50000000e+01, 8.60084106e+02, 3.06092232e-04, ...,
           1.06953388e-08, 8.95953534e-08, 1.18775756e-06],
          [5.00000000e+01, 6.02619812e+02, 9.80312347e-01, ...,
           2.79630513e-07, 1.30251522e-06, 8.41643669e-06]],

         [[5.60000000e+01, 1.21419846e+02, 1.94218475e+02, ...,
           3.49525362e-01, 4.96571362e-01, 8.22399259e-01],
          [4.40000000e+01, 1.09715843e+02, 2.13769394e+02, ...,
           1.79550871e-01, 2.68363863e-01, 4.77437854e-01],
          [6.70000000e+01, 1.41927917e+02, 1.68022537e+02, ...,
           5.70431650e-01, 9.10127878e-01, 1.66752565e+00],
...
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan]],

         [[           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          ...,
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan]]]]],
      dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>rabbits_res</span></div><div class='xr-var-dims'>(algorithm, chain, draw, rabbits_res_dim_0, rabbits_res_dim_1)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.0 -0.4418 1.351 ... nan nan nan</div><input id='attrs-d9cecbf4-ea0d-48f6-91c8-0194ff7c0ecb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d9cecbf4-ea0d-48f6-91c8-0194ff7c0ecb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f24f1f28-e25b-40d5-86c7-88204aeea2d1' class='xr-var-data-in' type='checkbox'><label for='data-f24f1f28-e25b-40d5-86c7-88204aeea2d1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[[ 0.0000000e+00, -4.4175312e-01,  1.3510162e+00, ...,
            2.9339402e+00, -6.8342769e-01, -8.8409048e-01],
          [ 0.0000000e+00,  5.3571004e-01,  6.9269067e-01, ...,
            1.9087670e+00, -5.2265859e-01, -6.9633925e-01],
          [ 6.1084723e-01, -2.1761173e-02,  7.3270410e-01, ...,
            5.4354572e-01, -9.6395206e-01, -5.3429329e-01],
          ...,
          [ 1.1817578e+00, -4.4251758e-01, -3.9846453e-01, ...,
           -7.9413934e-04, -2.7564554e-03, -9.2018209e-03],
          [-2.6967996e-01,  3.7221071e-01, -1.7466983e-02, ...,
           -1.0638599e-05, -8.5832718e-05, -8.0302334e-04],
          [-1.1313709e+00,  7.4873519e-01,  1.0298754e+00, ...,
           -2.4719647e-04, -8.5838384e-04, -2.7427420e-03]],

         [[ 0.0000000e+00, -3.1035709e-01,  1.5629425e+00, ...,
            2.7917008e+00, -7.0467752e-01, -9.0686184e-01],
          [ 0.0000000e+00,  5.0447661e-01,  6.3133150e-01, ...,
            1.9362288e+00, -5.1803750e-01, -6.9096804e-01],
          [ 6.1084723e-01, -7.7888884e-02,  6.9257981e-01, ...,
            5.6876135e-01, -9.5400572e-01, -5.1692981e-01],
...
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan]],

         [[           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          ...,
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan],
          [           nan,            nan,            nan, ...,
                      nan,            nan,            nan]]]]],
      dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>wolves</span></div><div class='xr-var-dims'>(algorithm, chain, draw, id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>8.0 13.98 54.63 ... nan nan nan</div><input id='attrs-393f30ea-c7fb-431b-a6df-a46e053cffd8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-393f30ea-c7fb-431b-a6df-a46e053cffd8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6137270d-ec5d-4ac3-a5b7-4ce3020b85ac' class='xr-var-data-in' type='checkbox'><label for='data-6137270d-ec5d-4ac3-a5b7-4ce3020b85ac' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[[  8.      ,  13.981825,  54.62884 , ...,  39.693707,
            29.51966 ,  22.001244],
          [  8.      ,  12.203411,  46.092064, ...,  44.539097,
            33.068485,  24.587189],
          [ 11.      ,  22.411224,  90.98986 , ...,  35.929466,
            26.81325 ,  20.115997],
          ...,
          [ 13.      , 220.06487 , 670.6752  , ...,  82.144585,
            60.854256,  45.081985],
          [  6.      , 631.91174 , 961.4004  , ..., 117.729645,
            87.216385,  64.61156 ],
          [  8.      ,  62.295723, 673.7098  , ...,  82.59386 ,
            61.187004,  45.328453]],

         [[  8.      ,  13.899094,  52.92574 , ...,  39.50916 ,
            29.39012 ,  21.912275],
          [  8.      ,  12.218336,  46.450695, ...,  44.62359 ,
            33.129883,  24.631401],
          [ 11.      ,  22.473158,  92.06322 , ...,  36.038357,
            26.890135,  20.168976],
...
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan],
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan],
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan]],

         [[       nan,        nan,        nan, ...,        nan,
                  nan,        nan],
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan],
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan],
          ...,
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan],
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan],
          [       nan,        nan,        nan, ...,        nan,
                  nan,        nan]]]]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>wolves_res</span></div><div class='xr-var-dims'>(algorithm, chain, draw, wolves_res_dim_0, wolves_res_dim_1)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.0 1.609 -2.656 ... nan nan nan</div><input id='attrs-6aebc186-20b3-46db-8a86-86cde2679bc0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6aebc186-20b3-46db-8a86-86cde2679bc0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d28b631f-bf77-4a66-be9c-614b6c403db0' class='xr-var-data-in' type='checkbox'><label for='data-d28b631f-bf77-4a66-be9c-614b6c403db0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[[ 0.00000000e+00,  1.60947001e+00, -2.65572834e+00, ...,
           -5.86275280e-01, -8.31859887e-01,  8.52513611e-01],
          [ 0.00000000e+00,  8.00549150e-01, -1.48650694e+00, ...,
            1.26778626e+00,  2.07485843e+00,  6.88268363e-01],
          [ 3.01511347e-01,  1.81425536e+00,  1.57357788e+00, ...,
           -1.48970449e+00,  4.22303319e-01,  1.97098255e-01],
          ...,
          [-2.77350068e-01,  1.20900834e+00,  5.14523864e-01, ...,
           -1.33996427e+00,  1.86829809e-02,  1.36725038e-01],
          [ 0.00000000e+00, -2.35172927e-01, -6.57940090e-01, ...,
            1.17079876e-01, -5.58561027e-01, -3.24896008e-01],
          [-7.07106709e-01, -1.68454587e+00, -7.97882676e-01, ...,
           -2.85411924e-01,  1.03934266e-01,  8.42395604e-01]],

         [[ 0.00000000e+00,  1.63644385e+00, -2.46401644e+00, ...,
           -5.58282673e-01, -8.09796095e-01,  8.73248577e-01],
          [ 0.00000000e+00,  7.95790195e-01, -1.53337741e+00, ...,
            1.25393713e+00,  2.06226778e+00,  6.78741932e-01],
          [ 3.01511347e-01,  1.79868937e+00,  1.45251107e+00, ...,
           -1.50559092e+00,  4.06872362e-01,  1.85042575e-01],
...
                       nan,             nan,             nan],
          [            nan,             nan,             nan, ...,
                       nan,             nan,             nan],
          [            nan,             nan,             nan, ...,
                       nan,             nan,             nan]],

         [[            nan,             nan,             nan, ...,
                       nan,             nan,             nan],
          [            nan,             nan,             nan, ...,
                       nan,             nan,             nan],
          [            nan,             nan,             nan, ...,
                       nan,             nan,             nan],
          ...,
          [            nan,             nan,             nan, ...,
                       nan,             nan,             nan],
          [            nan,             nan,             nan, ...,
                       nan,             nan,             nan],
          [            nan,             nan,             nan, ...,
                       nan,             nan,             nan]]]]],
      dtype=float32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6167600b-6e39-4b0f-8e5a-00db2d937089' class='xr-section-summary-in' type='checkbox'  ><label for='section-6167600b-6e39-4b0f-8e5a-00db2d937089' class='xr-section-summary' >Indexes: <span>(17)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-681d1eea-6be4-4191-b8e9-760e89a2c491' class='xr-index-data-in' type='checkbox'/><label for='index-681d1eea-6be4-4191-b8e9-760e89a2c491' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-f4502862-b209-46a1-a373-b95483134e42' class='xr-index-data-in' type='checkbox'/><label for='index-f4502862-b209-46a1-a373-b95483134e42' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
       ...
       1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=2000))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>alpha_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-afed9c21-e6ee-42fe-8447-8d01cf999b2e' class='xr-index-data-in' type='checkbox'/><label for='index-afed9c21-e6ee-42fe-8447-8d01cf999b2e' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
      dtype=&#x27;int64&#x27;, name=&#x27;alpha_dim_0&#x27;, length=120))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>alpha_normal_base_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-fe86ea38-3cc7-4b5c-a866-3e974c368bad' class='xr-index-data-in' type='checkbox'/><label for='index-fe86ea38-3cc7-4b5c-a866-3e974c368bad' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
      dtype=&#x27;int64&#x27;, name=&#x27;alpha_normal_base_dim_0&#x27;, length=120))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>alpha_species_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-caef5ad0-d44c-4b41-90e5-31515da29eff' class='xr-index-data-in' type='checkbox'/><label for='index-caef5ad0-d44c-4b41-90e5-31515da29eff' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1], dtype=&#x27;int64&#x27;, name=&#x27;alpha_species_dim_0&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>alpha_species_dim_1</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-62827e9c-2fbe-4cef-885c-7db684acbc63' class='xr-index-data-in' type='checkbox'/><label for='index-62827e9c-2fbe-4cef-885c-7db684acbc63' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2], dtype=&#x27;int64&#x27;, name=&#x27;alpha_species_dim_1&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>alpha_species_normal_base_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-d7fd37fc-23dc-43d3-b882-09e2730da5ad' class='xr-index-data-in' type='checkbox'/><label for='index-d7fd37fc-23dc-43d3-b882-09e2730da5ad' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1], dtype=&#x27;int64&#x27;, name=&#x27;alpha_species_normal_base_dim_0&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>alpha_species_normal_base_dim_1</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-cf7d97db-6465-4318-95cb-d9a5e1c56e18' class='xr-index-data-in' type='checkbox'/><label for='index-cf7d97db-6465-4318-95cb-d9a5e1c56e18' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2], dtype=&#x27;int64&#x27;, name=&#x27;alpha_species_normal_base_dim_1&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>id</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-07ea878e-ea16-49ce-b379-773ed8745413' class='xr-index-data-in' type='checkbox'/><label for='index-07ea878e-ea16-49ce-b379-773ed8745413' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
      dtype=&#x27;int64&#x27;, name=&#x27;id&#x27;, length=120))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-8f770d90-8b8b-465d-b6c3-877dbbcc55c8' class='xr-index-data-in' type='checkbox'/><label for='index-8f770d90-8b8b-465d-b6c3-877dbbcc55c8' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=&#x27;int64&#x27;, name=&#x27;time&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>rabbits_res_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-cddd7b3f-1e43-4aa4-9ff3-35690217d34f' class='xr-index-data-in' type='checkbox'/><label for='index-cddd7b3f-1e43-4aa4-9ff3-35690217d34f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
      dtype=&#x27;int64&#x27;, name=&#x27;rabbits_res_dim_0&#x27;, length=120))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>rabbits_res_dim_1</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-5ca0ff93-8aba-4b30-9b75-b969b509126f' class='xr-index-data-in' type='checkbox'/><label for='index-5ca0ff93-8aba-4b30-9b75-b969b509126f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=&#x27;int64&#x27;, name=&#x27;rabbits_res_dim_1&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>wolves_res_dim_0</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-2231e5a7-e084-4358-b806-390fd7a1b37c' class='xr-index-data-in' type='checkbox'/><label for='index-2231e5a7-e084-4358-b806-390fd7a1b37c' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
      dtype=&#x27;int64&#x27;, name=&#x27;wolves_res_dim_0&#x27;, length=120))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>wolves_res_dim_1</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-00df1043-c982-491f-8aad-faccfb232345' class='xr-index-data-in' type='checkbox'/><label for='index-00df1043-c982-491f-8aad-faccfb232345' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=&#x27;int64&#x27;, name=&#x27;wolves_res_dim_1&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>algorithm</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-5699fb5a-4ee0-4768-bc63-a098a74bd37a' class='xr-index-data-in' type='checkbox'/><label for='index-5699fb5a-4ee0-4768-bc63-a098a74bd37a' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;nuts&#x27;, &#x27;svi&#x27;], dtype=&#x27;object&#x27;, name=&#x27;algorithm&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>rabbit_species</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-6b751d7a-20fe-4c29-893c-22d6128cedc7' class='xr-index-data-in' type='checkbox'/><label for='index-6b751d7a-20fe-4c29-893c-22d6128cedc7' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;Cottontail&#x27;, &#x27;Jackrabbit&#x27;], dtype=&#x27;object&#x27;, name=&#x27;rabbit_species&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>experiment</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-25b277ea-9088-43d7-b4d1-5bd1abc6040d' class='xr-index-data-in' type='checkbox'/><label for='index-25b277ea-9088-43d7-b4d1-5bd1abc6040d' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;2010&#x27;, &#x27;2011&#x27;, &#x27;2012&#x27;], dtype=&#x27;object&#x27;, name=&#x27;experiment&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-9eb08280-752f-4f7d-8af7-7a9225f2bf04' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-9eb08280-752f-4f7d-8af7-7a9225f2bf04' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

az.plot_forest(
    data=[idata_nuts.posterior, idata_svi.posterior], 
    model_names=["NUTS", "SVI"],
    var_names=["beta"],
    ax=ax1,
    combined=True,
    hdi_prob=0.999
)
ax1.vlines(theta["beta"],*ax1.get_ylim(), color="black")

az.plot_forest(
    data=[idata_nuts.posterior, idata_svi.posterior], 
    model_names=["NUTS", "SVI"],
    var_names=["alpha_species"],
    ax=ax2,
    combined=True,
    hdi_prob=0.999
)
ax2.vlines(1,*ax2.get_ylim(), color="black")
ax2.vlines(3,*ax2.get_ylim(), color="black")

plt.tight_layout()
```


    
![png](hierarchical_model_files/hierarchical_model_36_0.png)
    


Both models fit a beta value very close to the true value, but are overly confident into their estimate, describing distribution that contain the true value only in the extreme tails of the distribution (0.999 HDI). In addition, the SVI model has problems to estimate the distributions of the alpha_species estimates. This uncertainty is much better covered by the NUTS algorithm.

### Hyper priors on species alpha and experimental variation estimate the variance of the parameter distribution accurately 

There are a few things we will work through to see what makes an unbiased fit of the parameters

+ Use hyperpriors for the hyperprior. Why? Do we really want to find out the alpha parameter for each species in each year, or do we want to find out the underlying alpha parameter for the species in any given year? By specifying hyper priors for the hyper prior, we can get both and on top of that may be able to better estimate the true variation in the data and get better parameter error estimates.
+ Normal prior for the alpha_species parameter. The data for the alpha species level is also drawn from a normal distribution with a single deviation parameter (sigma=0.1). Take a moment to think this through: This means, that the standard deviation of Cottontail is $N(1, 0.1)$ and Jackrabbit is $N(3, 0.1)$. If i now take a lognormal distribution with a constant deviation parameter I run into a problem, because in the lognormal case, the variance of the distribution scales with the scale of the parameter. So $Lognorm(3, 0.1)$ has a wider distribution than $Lognorm(1, 0.1)$. This becomes a real problem, because basically the distribution needs to fit 2 horses under the same roof. We get around this problem by using a normal distribution for the noise, or using different deviation parameters.

We take the liberty of using an unusual approach to specify our model parameters. This is no problem, because of the way the configuration backend is written. Because there are no interdependencies between the sections, we can safely specify our model parameters and then pass them to our configuration as a whole. This little trick will allow us to easily customize our entire posterior, and, more importantly, always specify in the correct order.


```python
# Level 1 Hyperpriors. These are supposed to converge on the true underlying patterns in the data
alpha_species_mu = Param(prior="halfnorm(scale=5)", dims=('rabbit_species',), hyper=True) # type: ignore
alpha_species_sigma = Param(prior="halfnorm(scale=5)", hyper=True) # type: ignore
alpha_sigma = Param(prior="halfnorm(scale=1)", hyper=True) # type: ignore

# Level 2 Hyperpriors
# Here we take the normal distribution in order to get the underlying variation structure right
# note that I also took the liberty of specifying the dimensional order differently, this makes it just a bit
# easier, because indexing of the hyperprior is not necessary.
alpha_species = Param(
    prior="norm(loc=[alpha_species_mu],scale=alpha_species_sigma)", # type: ignore
    hyper=True, dims=("experiment", "rabbit_species",)
) 

# Level 3 Model parameter priors
alpha = Param(prior="lognorm(s=alpha_sigma,scale=alpha_species[experiment_index, rabbit_species_index])", dims=("id",)) # type: ignore
beta = Param(prior="lognorm(s=1,scale=1)") # type: ignore



parameters = Modelparameters(
    alpha_species_mu=alpha_species_mu,
    alpha_species_sigma=alpha_species_sigma,
    alpha_species=alpha_species,
    alpha_sigma=alpha_sigma,

    alpha=alpha,
    beta=beta,

    **sim.config.model_parameters.fixed
)

sim.config.model_parameters = parameters

from pymob.sim.parameters import Expression
sim.config.error_model.wolves.obs_inv = Expression("res*jnp.sqrt(wolves+1e-06)+wolves")
sim.config.error_model.rabbits.obs_inv = Expression("res*jnp.sqrt(rabbits+1e-06)+rabbits")
sim.config.inference.n_predictions = 50
```


```python
sim.config.inference_numpyro.svi_iterations = 10_000
sim.config.inference_numpyro.svi_learning_rate = 0.0025
sim.dispatch_constructor()
sim.set_inferer("numpyro")

sim.inferer.run()
idata_svi_2 = sim.inferer.idata.copy()
```

    /home/flo-schu/miniconda3/envs/pymob/lib/python3.11/site-packages/pydantic/main.py:308: UserWarning: Pydantic serializer warnings:
      Expected `int` but got `float` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(


    Jax 64 bit mode: False
    Absolute tolerance: 1e-07
                           Trace Shapes:          
                            Param Sites:          
                           Sample Sites:          
       alpha_species_mu_normal_base dist       2 |
                                   value       2 |
    alpha_species_sigma_normal_base dist         |
                                   value         |
          alpha_species_normal_base dist   3   2 |
                                   value   3   2 |
            alpha_sigma_normal_base dist         |
                                   value         |
                  alpha_normal_base dist     120 |
                                   value     120 |
                   beta_normal_base dist         |
                                   value         |
                        rabbits_obs dist 120  12 |
                                   value 120  12 |
                         wolves_obs dist 120  12 |
                                   value 120  12 |


    100%|██████████| 10000/10000 [01:51<00:00, 89.70it/s, init loss: 95810480.0000, avg. loss [9501-10000]: 4121.1968]  
    arviz - WARNING - Shape validation failed: input_shape: (1, 2000), minimum_shape: (chains=2, draws=4)


                                      mean     sd  hdi_3%  hdi_97%  mcse_mean  \
    alpha[0]                         0.968  0.019   0.930    1.003        0.0   
    alpha[1]                         1.087  0.016   1.053    1.113        0.0   
    alpha[2]                         1.030  0.019   0.996    1.066        0.0   
    alpha[3]                         1.055  0.020   1.020    1.094        0.0   
    alpha[4]                         1.026  0.015   1.000    1.055        0.0   
    ...                                ...    ...     ...      ...        ...   
    alpha_species[2012, Jackrabbit]  2.937  0.019   2.903    2.972        0.0   
    alpha_species_mu[Cottontail]     0.925  0.012   0.904    0.947        0.0   
    alpha_species_mu[Jackrabbit]     2.886  0.020   2.848    2.924        0.0   
    alpha_species_sigma              0.114  0.008   0.099    0.129        0.0   
    beta                             0.018  0.000   0.018    0.018        0.0   
    
                                     mcse_sd  ess_bulk  ess_tail  r_hat  
    alpha[0]                             0.0    1874.0    1847.0    NaN  
    alpha[1]                             0.0    2056.0    2088.0    NaN  
    alpha[2]                             0.0    1970.0    1889.0    NaN  
    alpha[3]                             0.0    1742.0    1745.0    NaN  
    alpha[4]                             0.0    1902.0    2040.0    NaN  
    ...                                  ...       ...       ...    ...  
    alpha_species[2012, Jackrabbit]      0.0    1907.0    1738.0    NaN  
    alpha_species_mu[Cottontail]         0.0    1871.0    1851.0    NaN  
    alpha_species_mu[Jackrabbit]         0.0    1870.0    1769.0    NaN  
    alpha_species_sigma                  0.0    2167.0    1818.0    NaN  
    beta                                 0.0    2020.0    1719.0    NaN  
    
    [131 rows x 9 columns]



    
![png](hierarchical_model_files/hierarchical_model_40_4.png)
    



```python
sim.inferer.error_model = sim.inferer.parse_error_model(sim.config.error_model.all)
sim.posterior_predictive_checks()
```


    
![png](hierarchical_model_files/hierarchical_model_41_0.png)
    



```python
loglik, grad_loglik = sim.inferer.create_log_likelihood(return_type="joint-log-likelihood", check=False, vectorize=True, gradients=True)
```


```python
# TODO: Reactivate when everything is merged 

# sim.inferer.plot_likelihood_landscape(
#     ("alpha_species_mu", "beta"),
#     log_likelihood_func=loglik, 
#     gradient_func=grad_loglik
# )
```


```python
sim.config.model_parameters = parameters
sim.config.inference_numpyro.kernel = "nuts"
sim.config.inference_numpyro.nuts_max_tree_depth = 12
sim.dispatch_constructor()
sim.set_inferer("numpyro")

if False:
    sim.inferer.run()
    idata_nuts_2 = sim.inferer.idata.copy()
```

    Jax 64 bit mode: False
    Absolute tolerance: 1e-07


    /home/flo-schu/miniconda3/envs/pymob/lib/python3.11/site-packages/pydantic/main.py:308: UserWarning: Pydantic serializer warnings:
      Expected `int` but got `float` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(



```python
idata_svi_2.posterior.beta.mean(("chain", "draw"))
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
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;beta&#x27; ()&gt;
array(0.0175817, dtype=float32)</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'beta'</div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-62f8aa85-e119-4a15-afb8-82ad020915ee' class='xr-array-in' type='checkbox' checked><label for='section-62f8aa85-e119-4a15-afb8-82ad020915ee' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.01758</span></div><div class='xr-array-data'><pre>array(0.0175817, dtype=float32)</pre></div></div></li><li class='xr-section-item'><input id='section-bc0c0455-f1a4-4aab-97d0-5f1efa817da1' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-bc0c0455-f1a4-4aab-97d0-5f1efa817da1' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-bbf3a6b9-f042-48d0-b3bd-c8247b722278' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-bbf3a6b9-f042-48d0-b3bd-c8247b722278' class='xr-section-summary'  title='Expand/collapse section'>Indexes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-519506d8-274d-4434-8d1b-1d896224b399' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-519506d8-274d-4434-8d1b-1d896224b399' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
az.plot_forest(
    data=[idata_nuts.posterior, idata_svi.posterior, idata_svi_2.posterior], 
    model_names=["NUTS", "SVI", "SVI-hyper-hyper"],
    var_names=["beta"],
    ax=ax1,
    combined=True,
    hdi_prob=0.95
)
ax1.vlines(theta["beta"],*ax1.get_ylim(), color="black")

az.plot_forest(
    data=[idata_nuts.posterior, idata_svi.posterior, idata_svi_2.posterior], 
    model_names=["NUTS", "SVI", "SVI-hyper-hyper"],
    var_names=["alpha_species"],
    ax=ax2,
    combined=True,
    hdi_prob=0.95
)
ax2.vlines(1,*ax2.get_ylim(), color="black")
ax2.vlines(3,*ax2.get_ylim(), color="black")

plt.tight_layout()
```


    
![png](hierarchical_model_files/hierarchical_model_46_0.png)
    



**It seems the prior on $\sigma_{alpha}$ was missing**. If the sigma on alpha is included, the fits are slightly improved and the estimates for the species also become better. But also note, with three years, and some considerable variation it is not easy to get the estimate for the species right. I assume, that this model fitted with MCMC will perform better and include the true estimates with higher probability. Also, think it over! These priors describe the underlying relevant feats of the data. The expected growth rates of the rabbit species in general and their yearly variation.




```python
sim.config.case_study.scenario = "lotka_volterra_hierarchical_hyperpriors"
sim.config.create_directory("scenario")
sim.config.save()
```

    /home/flo-schu/miniconda3/envs/pymob/lib/python3.11/site-packages/pydantic/main.py:308: UserWarning: Pydantic serializer warnings:
      Expected `int` but got `float` - serialized value may not be as expected
      return self.__pydantic_serializer__.to_python(


    Scenario directory exists at '/home/flo-schu/projects/pymob/case_studies/lotka_volterra_case_study/scenarios/lotka_volterra_hierarchical_hyperpriors'.


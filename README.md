# Time Path

![build](https://github.com/flo-schu/timepath/actions/workflows/python-app.yml/badge.svg)

## current state

the model is currently calibrated on a dataset from chronic exposure experiments
of daphnia magna (OECD 212). Predictions of population trajectories is still
difficult, because starvation is not implemented correctly and microbial
feeding needs to be incorporated correctly into the model, to achieve the 
right level of carrying capacity. The next important steps are:

- bring the model to version 1.0
- develop case studies

To get back into the status of the project, check out the issues, particularly:

- <https://github.com/flo-schu/timepath/issues/40>
- <https://github.com/flo-schu/timepath/issues/38>

## Introduction

Timepath is a framework for modeling the life cycle of biological organisms as  individuals and populations. The model is designed to ensure consitency of basic thermodynamic principles; most fundamentally the conservation of mass and energy. Timepath addresses modelers who want increased flexibility in their biological systems model design while staying inside a reliable modeling framework. Timepath is not fast compared to methods based solely on differential equations or closed form solutions.

The package was originally designed to write a Python implementation of Dynamic Energy Budgets (DEB) - specifically DEBkiss (Jager, 2018). These models are designed to simulate the life cylce of the model laboratory organism Daphnia Magna. The goal was to analyze effects of exposure to toxicants under adverse environmental conditions of such organisms from laboratory experiments. It quickly became clear that typical lab experiments like OECD Test No. 211 _Daphnia magna Reproduction Test_ imposes adverse conditions on the test organism by the experimental design. Examples of such design flaws are:

- population pressure from offspring between exchange of experimental media,
- non-constant food conditions

To better utilize available data from animal testing, the goal was to incorporate exact laboratory conditions and experimental routines into the DEB model. This cannot be done with conventional approaches and thus the idea of timepath was born.

## Model description

Timepath models can be constructed from very low-level commands up to higher order functions that set up preconfigured experiments that can easily be modified with configuration files.

the simplest possible model

```python
from timepath.objects.organisms import Organism
from timepath.objects.environments import World
w = World()
o = Organism(X0={"Body":{"mass":1}}, environment=w)
CLOCK.dt = timedelta(1)
o.step()
w.step()
```

as can be seen the model acts via step functions that carry out basic routines. `Organism.step()` implements basic functions of life.

Timepath implements organisms as classes. Classes are lending themselves naturally to describe members of the phylogenetic tree of evolution. If one fundamental class has been written, the next description of a Genus, Order, Family, Species can always be built upon its ancestor. This in fact makes the definition of the Daphnia class, a thing of only a few lines of code.

```python
class Daphnia(Invertebrata):
    _ids = count(0)

    def __init__(
        self, 
        rate_uptake_esfenvalerate=0, 
        rate_elimination_esfenvalerate=0,
        **kwargs
        ):  
        super().__init__(**kwargs)

    def spawn_trigger(self):
        """
        For Daphnia reproduction takes place roughly every two days. This may
        be different in other species.
        """
        spawning_time = self.pools[ReproClock].time > self.spawning_interval
        return spawning_time and self.is_adult()
```

Also each organism needs an abiotic World to live in. This is implemented in the `World` class. Depending on what the goal of your model is, `World.step()` can contain different methods, implementing these.

### How are state variables tracked?

Timepath creates mass-pools with the `Pool` class. All these pools are state variables which are tracked and each `MassPool` is automatically contributing to the total mass balance of the system, which needs to stay constant

### How are experiments described?

Experiments can be described with a spreadsheet and a set of methods corresponding to the column names of the spreadsheet. For example the experiment procedure "feeding" can be described as follows:

```python
from timepath.sims.experiment import Experiment
from timepath.objects.base import external_environmental_event

class MyExperiment(Experiment)
    @external_environmental_event
    def feeding_mg(self, param):
        if param == 0:
            pass
        elif param > 0:
            self.flux(param, pool_in=self.env.pools[Food], pool_out=IO)
```

then the only thing needed is an experiment spread sheed listing the times and parameters for `feeding_mg`.

| time   | feeding_mg |
|--------|------------|
| 0 days |    1.0     |
| 1 days |    0.0     |
| 2 days |    2.0     |
| 4 days |    1.0     |
|--------|------------|

Internally at the beginning of the simulation this table is fed to the class `MyExperiment` and all feeding events are scheduled based on their time. When the runtime of the simulation has advanced to the event, it is executed. In the meantime, the organism or population does what it does. Feeding, Reproducing, Dying, ...

time can also be tracked as dates at arbitrary precision (e.g. `05.07.2022 13:42:05`). So, if observations in the experiment were done at different times on different dates and there is a doubt that it might have affected the results, this could also be modelled explicitely. Time in general in this framework is tracked in seconds and multiples

### This seems all very complicated

That is correct, because it is. Luckily, there is a very uncomplicated top-level function that does all the overhead work. The only thing you have to do is changing parameters of the configuration file and of course write your own functions and Organism classes, in case you want to.

```python
CFG1 = read_config("config/parameters/tests/daphnia.json")  # load config file
sim = basic_simulation(CFG1, logging_level_override="ERROR")  # init and run
sim.plot_life_history(store=False, show=True)  # plot the trajectories
```

templates of various config file can be found under <config/parameters/tests/>

## Data and parameter inference

Currently the model comes with two datasets, which are referenced in the publication ... Importing these data is very easy:

```python
from timepath.data.datasets import nanocosm, indy, to_xarray_dataset

individual_data = indy()  # dataset of multiple OECD 211 tests
population_data = nanocosm()  # dataset of a D. magna population experiment

# this is a wrapper to convert the data tuple to an xarray dataset with rich
# metadata information and clear labels.
xdata = to_xarray_dataset(individual_data)
```

__parameter inference__ is done via SBI (<https://github.com/mackelab/sbi>) this framework allows bayesian parameter inference from model simulations. The method belongs to approximate bayesian computation methods. But has the advantage that simulations can be easily parallelized upfront. The feature is still under active development and a generic method may be hard to obtain, since the optimization problem always depends on the model and its parameters.

### calibration of the model

- the model uses simulation based inference (SBI) for parameter estimation and inference on the model.
- to start out some 1000s of simulations should be generated (depending on the amount of parameters)
- for this use the script generate_sims.py
- if a High performance computation facility is available. This script is ready to be used on a cluster
- generate_sims.py draws parameters from a multiple independent distribution to get candidates that are later compared against model results

### calibrating the basic (control) organism

- the model rests on high quality calibration of parameters of the basic organism. For this about 1e6 simulations should be run and a neural network an estimator for the posterior density trained on input parameter samples and output summary statistics. A shell script that does all necessary involved steps from generating samples to training input can be executed on a SLURM scheduling
engine with (arguments denote arrays and batch size (n_sim for each job))

All relevant parameters are set in SBI Config files stored under config/parameters/sbi/

to test if everything works it is recommended to run `sbatch scripts/bash/run_tests.sh`
`scripts/bash/simulate_and_train.sh config/parameters/sbi/expo_control.json 1-500 200 1-20`

in detail the script executes the following steps (these can be run locally as well)

1. execute model/generate_sims.py
2. execute model/process_simulations.py (will generate 1 mio simulations)
3. execute model/train_network.py (twice 1 x SNLE 1x SNPE)
4. execute model/sbi_snle_sample_posterior.py will generate 10000 samples from the SNLE posterior

when everything is run, download the relevant files with `scp` from remote.

Then analyze with evaluate sbi. This will plot parameter estimates of all parameter clusters from not converged chains, as well as the plot of the zeroth sample (this should be replaced by plotting a group of samples) with high likelihood to also get some confidence on the trajectories

From there improper parameter sets can also be ruled out by visual inspection in the case there is more than 1 parameter cluster. Of course this can also be resolved by providing a) better summary statistics b) providing better prior information c) reducing the parameters of the model

Point c) is very important because equi-finality is a really good indicator for the model being not well informed on some parameters. If this is the case, the model may take shortcuts to get to the right goal.

Note that also only one parameter cluster with perfectly converged chains is no reliable indicator that the model was parameterized correctly. Although it is then likely. Good summary statistics are key to success.

The beauty of the approach is that the samples that come from the markov chains are already perfect samples with correct covariance enven though the covariance has not been established beforehand. So, if new parameters should be sampled or new models should be built on top of it, the posterior samples can just be re-used. It's just as good to make up very complicated distributions and sample again from them.

## Current state

### Biological systems and functions are built in and can be modelled

- Organisms:
  - Daphnia Magna
- Functions:
  - Survival
  - Growth
  - Reproduction
  - Feeding (functional response)
  - Toxicant uptake and elimination
  - Adverse Outcome Pathways (AOP). Very experimental so far
- Experiments:
  - OECD Test No. 211: Daphnia Magna Reproduction test
  - Population Development of Daphnia Magna

### Future features

- implement TKTD in the form of classical GUTS (General unified threshold )

### What timepath cannot do, yet

- simulate movement. This was up until now not necessary, but an implementation
  should be easily possible
- check units and ensure dimensionality consistency. A cause of frustration is
  oftentimes that errors result from conversion to milli, micro, etc.
  An experimental branch exists to utilize sympy for assertion of dimensional model consistency and conversion of units into SI system by default. However, this was not developed so far, but would be very desirable.
- Use symbolic math like sympy or other frameworks using equation graphs, for  
  sped up computation. This feature seemed very inspiring, but we realized that a lot of model freedom would have to be sacrificed to obtain not too impressive speedups.

## Installation  

```bash
pip install timepath
```

for a development version:

```bash
pip install git+https://github.com/flo-schu/timepath
```

for installing a development version and actively and iteratively update and
develop it it is recommended to download from source and install as editable

```bash
git clone git@github.com:flo-schu/timepath.git
cd timepath
pip install --editable .
```
### unit tests

it is recommended to run tests and see if everything works as expected in your
shell type `pytest` see <https://docs.pytest.org/en/7.1.x/> for more info

### General Notes

- Order of columns in events table determines execution order.
- Input of an event file always overrides the `time` configuration in `simulation` in the config.json file

### development set up

- Install Python >= 3.8
- vscode as a texteditor is recommended but not necessary scripts can be run from the commandline, line-by-line or in a debugging environment
- fork <https://github.com/flo-schu/timepath>
- git clone git@github.com:flo-schu/timepath.git
- `conda env create -f environment.yml`
- `conda activate timepath`
- for pyabc install do: `pip install --no-dependencies pyabc jabbar`
- alternatively `pip install -r requirements.txt` (there might still be some errors)
- install any missing packages with `pip install <package-name>`
- to test if everything works as expected run tests (very easy in vscode)
- add the file <config/settings.json> with the content
  `{"localhosts": ["YOURHOSTNAME"]}` where you replace `YOURHOSTNAME` by the actual name of your computer. On Windows you can get it with the command `hostname` in powershell. This file makes sure the data are correctly stored whether you are working remotely or not. This File must exist on the remote and locally, but always contain the local hostname. Also you can add the

### setting up remote computing

this assumes you are connecting to a linux remote server.
If your local machine is running on linux everything is very easy, if you are on windows, make sure you install windows subsystem linux (WSL) and enter the
WSL command line. Then also everything is easy

Create a new private-key, public-key pair
KEYCOMMENT and KEYNAME have to be replaced by names that describe the purpose
of the key, e.g. work_rita, work_rita_ed25519
Set a password to encrypt the key

```bash
ssh-keygen -t ed25519 -C KEYCOMMENT -f ~/.ssh/KEYNAME
```

then authorize the newly created key on the remote (for this)
USERNAME and REMOTE_SERVER refers to the address of the server and the username
of your account on that server

```bash
ssh-copy-id -i ~/.ssh/KEYNAME -o PreferredAuthentications=password -o PubkeyAuthentication=no USERNAME@REMOTE_SERVER
```

now you have a private key (and a public key) on your machine, the public key
as an authorized key on your remote and a password to decrypt the private key
all you need to do now is sign in and tell ssh to do it with the private 
key instead of a password

```bash
ssh -i ~/.ssh/KEYNAME USERNAME@REMOTE_SERVER
```

provide the details in `settings.json` to enable even more simple communication
with the remote server

### pyABC

#### single core sampling

to start sampling, run `tp-abc --case_study gutses --scenario pyabc_redis`

#### redis setup for Parallelization

##### usage

the absolute easiest way to use redis is:
`bash timepath/bash/abc_redis.sh -c gutses -s pyabc_redis -w 10 -p 1803 -a simulate`

where 
-c is the case study
-s scenario of the case study
-w is the number of workers
-p port of the server
-a password for the server

all other settings can be taken in the `abc.cfg` file in 
<case_studies/CASESTUDY/scenarios/SCENARIO/abc.cfg>

##### workflow

step 1:
start up redis server if needed add password and other arguments those can 
be also passed to the next call. By default password is `simulate` and port is
`1803`. These could be set with environmental variables
`redis-server config/redis_timepath.conf`

step 2:
if necessary adapt `abc.cfg` in the respective _scenario_ of the _case-study_
in the terminal to launch the sampler on the server 
`tp-abc --case_study gutses --scenario pyabc_redis` 

step 3:
launch any number of workers with (can also be done before step 2)
`sbatch -a 1-1 timepath/bash/abc_redis_worker.sh -p 1803 -a simulate`

##### explanations

for parallelization, `redis` is used. In the repository, a `redis_timepath.conf`
file is found, which sets some parameters for the redis server. For instance 
that it is started in the background (`deamonized yes`). It also loads another
`redis_password.conf` file, which you mast create. The file consists of only one
line: `requirepass YOURPASSWOR` where <YOURPASSWORD> is replaced by any password.
This is somewhat important for remote computing on clusters, since it ensures
that only persons that have the password can interact with the server. However,
since the server is set up only locally (localhost), there is no real threat by
not setting a password.

To start the server type into the terminal (in timepath root, where 
redis_timepath.conf is located):
`redis-server config/redis_timepath.conf`

in a testing environment run 
`redis-server redis_timepath.conf --port 1111 --requirepass test`

The arguments `--port` and `--requirepass` overwrite the arguments set in the
.conf file

no output will be returned from this command. However, to check if the server
is running call: `pgrep redis-server`. This will list the process-id of the
redis-server. 

To log into the server, call: `redis-cli -p PORT -a PASSWORD`. Here <PORT> and 
<PASSWORD> are replaced by your values. PORT can be found in the redis_timepath.conf
file and defaults to 1803.

To test if the server responds, type `PING`. The server shout respond <PONG>

To terminate the server log into it via `redis-cli` as shown above and type
`shutdown`



#### pyabc with redis

redis is principally used for pyABC <https://pyabc.readthedocs.io/>
How this is done is described in <https://pyabc.readthedocs.io/en/latest/sampler.html>

currently I use the file <case_studies/tktd_mix/scripts/abc.py> to test this 
feature. Running this script, results in launching the sampler, which is connected
to a redis server `python case_studies/tktd_mix/scripts/abc.py`

To do the actual sampling, redis-workers need to be started. In this example,
the worker is started on the testing server

first set the PYTHONPATH environmental variable with `case_studies`, 
Because, part of pyabc's usage of redis is to pickle the model to memory and
unpickle it on every worker. Therefore any packages not inside the system path
are not accessible to the redis worker and have to be made accessible. This can 
be done by setting the PYTHONPATH environmental variable
`export PYTHONPATH=~/projects/timepath/case_studies`

`abc-redis-worker --port=1111 --password=test`

##### to run the whole thing

start the server (make sure server is not running)
`bash case_studies/tktd_mix/scripts/abc_redis_server.sh -c case_studies/tktd_mix/scenarios/diuron_pyabc/abc.cfg -p 1803 -a simulate`

start the workers
`sbatch -a 1-200 case_studies/tktd_mix/scripts/abc_redis_worker.sh -p 1803 -a simulate`

check the status
`abc-redis-manager info --port 1803 --password simulate`

if the sampler shows a low number of workers some of them may have been caught
up. Especially if the number of accepted samples is greater than the necessary population, workers can be reset

`abc-redis-manager reset-workers --port 1803 --password simulate`

If workers get stuck, this can have to do with an out of memory error. 
Test this with `sacct -j <job_id>`

#### evaluate pymc run

run the following scripts

`python case_studies/tktd_mix/scripts/abc_eval.py --case_study tktd_mix --scenario diuron_pyabc_asd`

`python model/posterior_predictions.py --case_study tktd_mix diuron_pyabc_asd -n 10`

`python model/plot_posterior_predictions.py --case_study tktd_mix diuron_pyabc_asd`


### PYMC and JAX

for running bayesian inference on timeseries or alike install pymc
additionally install jax for increased performance.

On linux just pip install jaxlib; pip install jax
On windows download a prebuilt wheel on <https://whls.blob.core.windows.net/unstable/index.html> and install with `pip install <version_downloaded_wheel.whl>`
then `pip install jax==version` where version is replaced with the same version
as jaxlib.

Make sure to choose the same python version as running on your system/repository

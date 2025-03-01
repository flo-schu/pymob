# GUTS-RNA-pulse substance specific

## Problem formulation

Risk assessment of chemicals needs to get away from making retrospective assessments after evaluating the effects of chemicals on ecosystems or on individuals to derive safe test concentrations. This need is one hand mandated by the desire for healthy ecosystems, the precautionary principle and ethical considerations to reduce animal testing.

With this model we investigate the potential to integrate omics data into the existing general unified threshold model for survival (GUTS) [Jager.2011, Jager.2018], which is considered ready for use in environmental risk assessment [EFSA.2018].

The growing availability of 'omics data drives the abstraction of bio-physical insights into the processes that govern molecular responses to changing environments.
The integration of 'omics data into mechanistic models therefore offers a promising solution for advancing risk assessment for chemicals and chemical mixtures, because in theory it can connect the cellular processes induced after toxicant exposure with observed effects in the organism [Perkins.2019,Murphy.2018].
Integrating such approaches into existing mechanistic modeling frameworks envisions the prediction of toxicant effects for untested species--substance combinations and mixtures as a very desirable long-term goal for a predictive environmental risk assessment.

The GUTS-RNA-pulse model is the first approach to integrate time-resolved gene-expression data (Nrf2 fold-change) into a GUTS model. 

### Target audience | model aims

It is aimed at researchers in the field of environmental risk assessment. It is in an early stage of scientific development and serves as a proof of principle. The model is designated to investigate the process dynamics and the interplay between chemical uptake and metabolization and their interactions with gene-expression dynamics. 

The aim of the model is to obtain a better understanding on modeling the temporal dynamics from exposure to effect with multiple intermediate steps. Incorporation of intermediate steps in the model fitting, forces the modellers to incorporate more realistic assumptions of biological processes and step-by-step move to better approximations of the reality.

### Operability range

It can be used to extrapolate internal concentrations, Nrf2 fold-changes, survival rates forward in time and for untested concentrations. The model it is not ready to be used outside of the range of the calibrated chemicals and should not be used for predictions of untested chemicals. 


## Model description

### Synopsis

![image-5.png](tktd_rna_3_6c_substance_specific_files/image-5.png)

Figure 1. Graphical model description. Created with [Biorender.com](https://biorender.com)

The model RNA-pulse describes the damage dynamic as a gene-expression pulse that is calibrated on *nrf2* expression data. It uses a sigmoid function to model the threshold dependent activation of *nrf2* expression and a concentration dependent exponential decay of *nrf2* molecules. Coupled with active metabolization of the internal concentration of the chemical this leads to a pulse like behavior. In addition *nrf2* serves as a proxy for toxicodynamic damage in the standard GUTS survival model of stochastic death.

### Detailed model description

$$
\begin{align}
\frac{dC_i}{dt} &= k_i~C_e - k_m~C_i~P \\
\frac{dR}{dt} &= r_{rt}~\text{activation}(C_i,~C_{i,max},~ z_{ci},~ v_{rt}) - k_{rd} ~ (R - R_0) \\
\frac{dP}{dt} &= k_p~ ((R - R_0) - P) \\
h(t) &= k_k~ max(0, R(t) - z) +  h_b \\
S(t) &= e^{-\int_0^t h(t) dt}
\end{align}
$$

#### Uptake and elimination kinetics

Uptake and elimination kinetics (Eq. 1) are determined by the uptake rate constant $k_i$, which exclusively depends on the external concentration $C_e$. Although external concentrations were measured for those datasets where internal concentrations were available, a we chose not to model uptake from the environment and decay, because the observed environmental concentrations were relatively stable and Diuron and Naproxen are known to be stable compounds [Giacomazzi.2004,Wojcieszynska.2020] and Diclofenac has a DT50 of 8 days, but was also observed stable in the experiments.

![Diclofenac_external.png](tktd_rna_3_6c_substance_specific_files/image-2.png)

![Diuron_external.png](tktd_rna_3_6c_substance_specific_files/image-3.png)

![Naproxen_external.png](tktd_rna_3_6c_substance_specific_files/image.png)

Figure 2. Environmental concentrations over time for the assessed chemicals Diclofenach, Diuron and Naproxen.  

The latter (detoxification) term $k_m~C_i~P$ can be understood as a Michaelis-Menten enzyme kinetic for relatively low substrate concentrations $C_i$ (Fig. 1). Passive chemical decay, independent of the $P$ concentration, is not considered in this model.

![image-4.png](tktd_rna_3_6c_substance_specific_files/image-4.png)

Figure 3. Schematic of enzyme kinetics at the example of Michaelis-Menten kinetics https://en.wikipedia.org/wiki/Enzyme_kinetics#/media/File:KinEnzymo(en).svg

Michaelis-Menten kinetics is defined by the following term. $v = \frac{dp}{dt} = \frac{k_{cat}~E~S}{K_m + S}$. For low S, the equation is governed by the numerator and $\frac{k_{cat}}{K_m}$ is known as the specificity constant that explains how efficiently the substrate is converted into the product. Using the full equation would be more desirable, to describe also the saturated phase of the enzymatic metabolization of the chemical. In this model, we resort to the most simple equation, where the decay linearly depends on the product of enzyme and substrate. The most important reason for this decision is that neither data for the enzyme(s) nor for the reaction product is available.

With further data availability, such as measured metabolites of the chemical, more advanced enzyme kinetics could be described. Another limitation of this model is that in reality certainly more than one enzyme will be responsible for the degradation. It is highly recommended that a better resolved elimination process (passive and active elimination) is investigated in depth in future versions of the model.

#### Gene expression and decay

While internal concentrations varied over 3 orders of magnitude between the different substances, *nrf2* expression varied within 1 order of magnitude. This observations are in line with the understanding that gene-transcription follows zero order kinetics [Qiu.2022,Xu.2023a], which effectively decouples it from the magnitude of the internal concentrations. In addition, *nrf2* expression showed pulse like patterns, which elicits the inclusion of a deactivation of expression (Eq. 2). This combined process of *nrf2* transcription, we therefore chose to model as a zero order kinetic, activated by a concentration dependent sigmoid function.

$$activation(C_i,~C_{i,max},~z_{ci}, ~v_{rt}) = 0.5 + \frac{1}{\pi} ~ arctan(v_{rt} ~ (\frac{C_i}{C_{i,max}} - z_{ci}))$$

where the threshold $z_{ci}$ and slope $v_{rt}$ are fittable parameters. This process is also illustrated in the graphical model description and understands that there is a maximum cellular capacity for expression of the *nrf2* gene. The activation of expression however is concentration dependent.

RNA decay is modeled as a first-order kinetic decay equation, which is in agreement with concentionally applied RNA decay models [Chen.2008,Blake.2024].

With the assumptions of a threshold-triggered activation of 0th-order gene-transcription (expression) and 1st order RNA-decay, the model is capable of increasing RNA-concentration after exposure to toxicants and depleting it again after the concentration falls below a threshold. Because the concentration of RNA-expression is indirectly coupled to the elimination of the compound, various patterns of internal concentration dynamics and gene-expression dynamics under constant exposure concentrations can be modeled.

The model is limited by describing only *nrf2* as a driver of the stress response. In reality, other genes also contribute to the stress response. Other important genes in the stress response are CYP1A, CYP1B, CYP1C, GST and UGT.

Due to experimental constraints it is not possible to measure the absolute copy number of RNA transcripts in an organism and give a concentration. Instead, the gene-expression is always given relative to the control organism as a multiple (fold-change) which is 1 in an unchanged state. This requires modeling the gene-expression system also in fold-change units relative to the basline of $R_0 = 1$.

#### Protein dynamics

On a biological level, the metabolization of a chemical is not driven by the concentration of *nrf2* transcripts, but by downstream products. After transcription (and activation), *nrf2* dissociates into the nucleus and activates the expression of other genes, so called antioxidant response elements (ARE). These genes are translated to proteins that handle the stress response, including enzymes like catalases, gluthathionperoxidases, and peroxidates.
All processes involved in the active detoxification rate are aggregated into a single quantity $P$, which changes depending on the *nrf2* concentration and the metabolizing protein concentration with a dominant rate constant $k_p$ (Eq. 3). This equation is included to describe the process that metabolization can persist after the transient gene-regulation pulse has passed. This decision is based on insights that proteins have half-lifes of 20-46 hours [Harper.2016], while *nrf2* RNA transcripts have approximated half-lifes of only 20 minutes [Kobayashi.2004].

#### Survival functions

The survival probability $S$ (Eq. 5) is modeled according to the stochastic death assumption of the GUTS framework [Jager.2011,Jager.2018], where the hazard is approximated by nrf2 fold-change (Eq. 4). The given equations are standard textbook functions of the survival analysis and will not be explained in detail.

### Model parameters

TKTD Parameters used in the GUTS-RNA-pulse model. The column `Assumed substance independence` indicates whether a parameter is supposed to be shared for multiple  substances.
| Parameter              | Definition | Unit | Assumed substance independence  |
|------------------------|------------|------|---------------------------------|
| ${k}_{i}$              | Uptake rate constant of the chemical into the internal compartment of the ZFE | $h^{-1}$ | no  |
| ${k}_{m}$              | Metabolization rate constant from the internal compartment of the ZFE| $\frac{L}{\mu mol~h}$ | no  |
| ${z}_{\text{ci}}$      | Scaled internal concentration threshold for the activation of *nrf2* expression | $\frac{\mu mol~L^{-1}}{\mu mol~L^{-1}}$ | no  |
| ${v}_{\text{rt}}$      | Scaled responsiveness of the *nrf2* activation (slope of the activation function) | $\frac{\mu mol~L^{-1}}{\mu mol~L^{-1}}$ | yes/no $^a$  |
| ${r}_{\text{rt}}$      | Constant *nrf2* expression rate after activation $^b$ | fc $^c$ | yes  |
| ${k}_{\text{rd}}$      | Nrf2 decay rate constant | $h^{-1}$ | yes  |
| ${k}_{p}$              | Dominant rate constant of synthesis and decay of metabolizing proteins | $h^{-1}$ | yes  |
| ${z}$                  | Effect *nrf2*-threshold of the hazard function $^b$ | fc $^c$ | yes  |
| ${k}_{k}$              | killing rate constant for *nrf2* $^b$ | $fc^{-1}~h^{-1}$ $^c$ | yes |
| ${h}_{b}$              | background hazard rate constant | $h^{-1}$ | yes  |
| $\sigma_{\text{cint}}$ | Log-normal error of the internal concentration | | yes  |
| $\sigma_{nrf2}$        | Log-normal error of the *nrf2* expression $^b$ | | yes  |

a: In an unscaled version of the activation function, $v_{rt}$ is not considered substance independent, due to an inverse relationship between $v_{rt}$ and $C_{i,max}$ 

b: relative to the *nrf2* concentration in untreated ZFE (fold-change)

c: fold change: $\frac{\mu mol~Nrf2\text{-treatment}~L^{-1}}{\mu mol~Nrf2\text{-control}~L^{-1}}$ 

### Caveats 💥

1. When calculating treatment effects it should be made sure that effects are calculated differentially to the initial value of the RNA expression
2. When $R_0 \neq 1$, the RNA expression has to be divided by the baseline to obtain fold-change values, after the ODE has been solved.


## Implementation

For the implementation of the model the package `pymob` was developed. Pymob is publically available on github https://github.com/flo-schu/pymob and deployed on PyPi https://pypi.org/project/pymob/. The documentation for pymob is available on https://pymob.readthedocs.io/en/latest/
Parameter recover of implemented inference methods are tested on the Lotka-Volterra model as a basic example, and is included in the repository. 

The implementation describes case studies (models with datasets) and scenarios (parameterizations, exposures, inference-procedures). Case studies are self-contained building blocks and describe the simulation (sim.py) deterministic model (mod.py), proabilistic error models (prob.py), plotting functions (plot.py) and datasets (data.py) in a modular way. Scenarios are given in config files. The data and results are traced with `datalad` so that data and results can be traced and reproduced for developing versions of the model.

In the following the GUTS-RNA-pulse (3.6c) model is fitted to data and analyzed with the `pymob` framework.


```python
import os
import json
import warnings
from functools import partial

import numpy as np
import arviz as az
import matplotlib as mpl
from matplotlib import pyplot as plt

from pymob import Config
from tktd_rna_pulse.sim import SingleSubstanceSim3

# Ignore warnings and change working directory to the root of the case study
warnings.filterwarnings("ignore")
```

The **substance-specific** scenario is loaded for the case study **tktd-rna-pulse** 
The model described above is specified in the pymob framework in the function `tktd_rna_3_6c` in the module `case_studies.tktd_rna_pulse.mod`


```python
# initialize the case study and insert the model
config = Config("../scenarios/rna_pulse_5_substance_specific/settings.cfg")
# change the package directory, because working in a jupyter notebook sets the root to the folder of the working directory
# the package gives the base directory of the case-study
config.case_study.package = "../.."

sim = SingleSubstanceSim3(config)
sim.setup()
```

    MinMaxScaler(variable=cint, min=0.0, max=6364.836264471382)
    MinMaxScaler(variable=nrf2, min=0.0, max=3.806557074337876)
    MinMaxScaler(variable=survival, min=0.0, max=18.0)
    Results directory exists at '/home/flo-schu/projects/pymob/case_studies/tktd_rna_pulse/results/rna_pulse_5_substance_specific'.
    Scenario directory exists at '/home/flo-schu/projects/pymob/case_studies/tktd_rna_pulse/scenarios/rna_pulse_5_substance_specific'.


## Solving the model with the autodifferentiation frameworks JAX and diffrax

JAX (http://jax.readthedocs.io/) is a autodifferentiation framwork that just-in-time (jit) compiles python code (~ 10s) into highly efficient numerical expressions. Diffrax (https://docs.kidger.site/diffrax/) is built on JAX and leverages this platform to provide solutions to ordinary differential equation (ODE) systems, including their sensitivities with respect to the parameters. In mod.py the diffrax based solver is described and handles the indexing and broadcasting of parameters from the model fitting to the complex case of fitting models to scattered data structures.

For each model evaluation (for all experiments) `pymob` dispatches an evaluator with a given set of parameter values and other optional arguments, that need to be specified in the solver. The evaluator stores the simulation output in its native form (here a dictionary) and can convert dictionaries to numpy arrays, based on the provided data dimensionality in the configuration file.


```python
from dataclasses import dataclass
from pymob.solvers.diffrax import JaxSolver

@dataclass(frozen=True)
class Solver(JaxSolver):
    rtol: float = 1e-3
    atol: float = 1e-6
    batch_dimension: str = "id"
```


```python
sim.solver = Solver
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
sim.dispatch_constructor()
evaluator = sim.dispatch(theta={})
evaluator()

evaluator.results
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
Dimensions:          (id: 202, time: 23)
Coordinates:
  * id               (id) object &#x27;101_0&#x27; &#x27;101_1&#x27; &#x27;106_0&#x27; ... &#x27;66_4&#x27; &#x27;66_5&#x27; &#x27;6_0&#x27;
  * time             (time) float64 24.0 25.5 27.0 30.0 ... 114.0 117.0 120.0
    hpf              (id) float64 24.0 24.0 24.0 24.0 ... 24.0 24.0 24.0 24.0
    nzfe             (id) float64 nan nan nan nan nan ... 9.0 9.0 9.0 9.0 20.0
    treatment_id     (id) int64 101 101 106 106 112 112 118 ... 66 66 66 66 66 6
    experiment_id    (id) int64 36 36 36 36 36 36 36 36 ... 27 27 27 27 27 27 1
    substance        (id) &lt;U10 &#x27;diuron&#x27; &#x27;diuron&#x27; ... &#x27;naproxen&#x27; &#x27;naproxen&#x27;
    substance_index  (id) int64 0 0 0 0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2 2 2 2
Data variables:
    cext             (id, time) float32 2.34 2.34 2.34 ... 349.5 349.5 349.5
    cint             (id, time) float32 0.0 1.755 3.51 ... 1.502e+04 1.546e+04
    nrf2             (id, time) float32 1.0 1.028 1.042 ... 1.199 1.2 1.199
    P                (id, time) float32 0.0 0.001166 0.003685 ... 0.1966 0.1972
    H                (id, time) float32 0.0 0.0004788 0.001558 ... 0.3338 0.3459
    survival         (id, time) float32 1.0 0.9995 0.9984 ... 0.7162 0.7076</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-cddf01cf-deaf-421e-bf8c-70a86247aa1a' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cddf01cf-deaf-421e-bf8c-70a86247aa1a' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>id</span>: 202</li><li><span class='xr-has-index'>time</span>: 23</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-20f4beae-bca3-4dd8-9770-ab619ad00c85' class='xr-section-summary-in' type='checkbox'  checked><label for='section-20f4beae-bca3-4dd8-9770-ab619ad00c85' class='xr-section-summary' >Coordinates: <span>(8)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>id</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;101_0&#x27; &#x27;101_1&#x27; ... &#x27;66_5&#x27; &#x27;6_0&#x27;</div><input id='attrs-332a2228-75be-4577-9081-e8657bea2cde' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-332a2228-75be-4577-9081-e8657bea2cde' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2db07393-f567-46dd-aaf1-c13bca5b012b' class='xr-var-data-in' type='checkbox'><label for='data-2db07393-f567-46dd-aaf1-c13bca5b012b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;101_0&#x27;, &#x27;101_1&#x27;, &#x27;106_0&#x27;, ..., &#x27;66_4&#x27;, &#x27;66_5&#x27;, &#x27;6_0&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>24.0 25.5 27.0 ... 117.0 120.0</div><input id='attrs-dd82493f-ad83-4c4c-9d43-2c99d404a20a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dd82493f-ad83-4c4c-9d43-2c99d404a20a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f5c7e975-5d6a-4aab-b10c-de9f897842de' class='xr-var-data-in' type='checkbox'><label for='data-f5c7e975-5d6a-4aab-b10c-de9f897842de' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 24. ,  25.5,  27. ,  30. ,  32. ,  33. ,  34. ,  36. ,  48. ,  54. ,
        60. ,  72. ,  74. ,  75. ,  78. ,  81. ,  84. ,  96. , 104. , 108. ,
       114. , 117. , 120. ])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>hpf</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>24.0 24.0 24.0 ... 24.0 24.0 24.0</div><input id='attrs-732d15f7-caa5-492c-a748-4e44e742c8e0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-732d15f7-caa5-492c-a748-4e44e742c8e0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8f8f5331-0e1e-492f-a423-df12ab142cda' class='xr-var-data-in' type='checkbox'><label for='data-8f8f5331-0e1e-492f-a423-df12ab142cda' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
       24., 24., 24., 24., 24., 24., 24.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>nzfe</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>nan nan nan nan ... 9.0 9.0 20.0</div><input id='attrs-4c6726ce-83c7-4611-841f-ce0adbac7d74' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4c6726ce-83c7-4611-841f-ce0adbac7d74' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e6119061-21ca-49c1-a784-2369b96224d1' class='xr-var-data-in' type='checkbox'><label for='data-e6119061-21ca-49c1-a784-2369b96224d1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9., 20.,
        2.,  2.,  2.,  9.,  9.,  9., 18., 18.,  8.,  9.,  9.,  9., 20.,
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9.,  9., 18.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9., 20., nan, nan, nan, nan, nan, nan,
       nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 18., 18., 18.,
       18., 18., 18., 18., 18., 18., 18., 18., 18., 18., 18., 18., 18.,
       18., 18., 18., 18., 18., 18., 18., 18., 18., 18., 12., 20., 20.,
       20.,  9.,  9.,  9.,  9.,  9.,  9., 12.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9., 20.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>treatment_id</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>101 101 106 106 112 ... 66 66 66 6</div><input id='attrs-ef0dd89b-b28f-4257-a578-d7a511b763f3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ef0dd89b-b28f-4257-a578-d7a511b763f3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3641622f-67c2-4f43-ad1e-1aac42953b59' class='xr-var-data-in' type='checkbox'><label for='data-3641622f-67c2-4f43-ad1e-1aac42953b59' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([101, 101, 106, 106, 112, 112, 118, 118, 124, 124, 193, 194, 195,
       196, 197, 198, 199, 200, 201, 202, 203, 204, 206, 207, 208, 209,
       210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,  42,
        51,  51,  51,  52,  52,  52,  53,  53,  69,  70,  70,  70,  10,
       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
        38,  54,  54,  55,  55,  55,  56,  56,  56,  56,  56,  56,  57,
        57,  57,  57,  57,  57,  62,  62,  67,  67,  67,  67,  67,  67,
        68,  68,  68,  68,  78,  78,  80,  82,  82,  84,  84,  86,  86,
        88,  88,  88,  90,  90,  91,  91,  92,  92,  93,  93,  94,  94,
        95, 102, 102, 107, 113, 113, 119, 119, 125, 125, 233, 234, 235,
       236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 247, 249, 251,
       253, 254, 255, 256, 257, 258, 259, 260, 261, 262,  27,  28,  33,
        40,  58,  58,  58,  59,  59,  59,   5,  60,  60,  61,  61,  63,
        63,  63,  63,  64,  64,  64,  64,  65,  65,  65,  65,  65,  65,
        66,  66,  66,  66,  66,  66,   6])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>experiment_id</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>36 36 36 36 36 36 ... 27 27 27 27 1</div><input id='attrs-09467068-8c29-4c39-bb03-551efcd1c100' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-09467068-8c29-4c39-bb03-551efcd1c100' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ffd61a90-268f-4de8-b6e7-225b990eb967' class='xr-var-data-in' type='checkbox'><label for='data-ffd61a90-268f-4de8-b6e7-225b990eb967' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 43, 43, 43, 43, 43, 43, 43,
       43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44,
       44, 44, 44, 44, 14, 19, 19, 19, 20, 20, 20, 20, 20, 30, 30, 30, 30,
        3, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41,
       41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 12, 21, 21, 21, 21, 21, 22,
       22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 25, 25, 28, 28, 28, 28,
       28, 28, 29, 29, 29, 29, 32, 32, 33, 34, 34, 34, 34, 34, 34, 34, 34,
       34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36,
       36, 36, 36, 36, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48,
       48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49,  7,  7, 10, 13,
       23, 23, 23, 23, 23, 23,  1, 24, 24, 24, 24, 26, 26, 26, 26, 26, 26,
       26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,  1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>substance</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>&lt;U10</div><div class='xr-var-preview xr-preview'>&#x27;diuron&#x27; &#x27;diuron&#x27; ... &#x27;naproxen&#x27;</div><input id='attrs-04e85b3b-2edd-473c-844b-ffd8281d23ce' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-04e85b3b-2edd-473c-844b-ffd8281d23ce' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a43c3a0b-c168-4392-90cd-765d3e5717fe' class='xr-var-data-in' type='checkbox'><label for='data-a43c3a0b-c168-4392-90cd-765d3e5717fe' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;diuron&#x27;, &#x27;diuron&#x27;, &#x27;diuron&#x27;, ..., &#x27;naproxen&#x27;, &#x27;naproxen&#x27;, &#x27;naproxen&#x27;],
      dtype=&#x27;&lt;U10&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>substance_index</span></div><div class='xr-var-dims'>(id)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2</div><input id='attrs-72620176-6461-47de-9ed8-0a5c97fba05d' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-72620176-6461-47de-9ed8-0a5c97fba05d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2167ae48-98c4-4573-b716-7ff079216fc8' class='xr-var-data-in' type='checkbox'><label for='data-2167ae48-98c4-4573-b716-7ff079216fc8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>cext_diuron :</span></dt><dd>µmol L-1</dd><dt><span>cint_diuron :</span></dt><dd>µmol L-1</dd><dt><span>nrf2 :</span></dt><dd>fold-change</dd><dt><span>lethality :</span></dt><dd>count</dd><dt><span>substance :</span></dt><dd>[&#x27;diuron&#x27;, &#x27;diclofenac&#x27;, &#x27;naproxen&#x27;]</dd><dt><span>ids_subset :</span></dt><dd>[]</dd><dt><span>excluded_experiments :</span></dt><dd>[15 16 18 31 42 45 46  2 37 38 39]</dd></dl></div><div class='xr-var-data'><pre>array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-01b65fa0-68cc-45c4-9a9c-f9f17fea9631' class='xr-section-summary-in' type='checkbox'  checked><label for='section-01b65fa0-68cc-45c4-9a9c-f9f17fea9631' class='xr-section-summary' >Data variables: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>cext</span></div><div class='xr-var-dims'>(id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>2.34 2.34 2.34 ... 349.5 349.5</div><input id='attrs-8c7abc14-0c98-499c-aaff-2bc01f6cc56f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8c7abc14-0c98-499c-aaff-2bc01f6cc56f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bc8904ee-1e10-4c25-9d53-915218fd1cb9' class='xr-var-data-in' type='checkbox'><label for='data-bc8904ee-1e10-4c25-9d53-915218fd1cb9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[  2.34  ,   2.34  ,   2.34  , ...,   2.34  ,   2.34  ,   2.34  ],
       [  2.34  ,   2.34  ,   2.34  , ...,   2.34  ,   2.34  ,   2.34  ],
       [  5.16  ,   5.16  ,   5.16  , ...,   5.16  ,   5.16  ,   5.16  ],
       ...,
       [309.2293, 309.2293, 309.2293, ..., 309.2293, 309.2293, 309.2293],
       [309.2293, 309.2293, 309.2293, ..., 309.2293, 309.2293, 309.2293],
       [349.5388, 349.5388, 349.5388, ..., 349.5388, 349.5388, 349.5388]],
      dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>cint</span></div><div class='xr-var-dims'>(id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.0 1.755 ... 1.502e+04 1.546e+04</div><input id='attrs-88595590-d104-4819-929b-69de3ecfd874' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-88595590-d104-4819-929b-69de3ecfd874' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cd6bd683-f264-42dd-92d7-4d467875e5dd' class='xr-var-data-in' type='checkbox'><label for='data-cd6bd683-f264-42dd-92d7-4d467875e5dd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.00000000e+00, 1.75499213e+00, 3.50989342e+00, ...,
        1.02959900e+02, 1.06292755e+02, 1.09619644e+02],
       [0.00000000e+00, 1.75499213e+00, 3.50989342e+00, ...,
        1.02959900e+02, 1.06292755e+02, 1.09619644e+02],
       [0.00000000e+00, 3.86998129e+00, 7.73976469e+00, ...,
        2.26806412e+02, 2.34128525e+02, 2.41434387e+02],
       ...,
       [0.00000000e+00, 2.31920868e+02, 4.63829010e+02, ...,
        1.29151836e+04, 1.33019043e+04, 1.36862002e+04],
       [0.00000000e+00, 2.31920868e+02, 4.63829010e+02, ...,
        1.29151836e+04, 1.33019043e+04, 1.36862002e+04],
       [0.00000000e+00, 2.62152863e+02, 5.24290283e+02, ...,
        1.45877383e+04, 1.50247852e+04, 1.54591396e+04]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>nrf2</span></div><div class='xr-var-dims'>(id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>1.0 1.028 1.042 ... 1.199 1.2 1.199</div><input id='attrs-8cf8575d-332b-4144-aaa2-51130c04d41a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8cf8575d-332b-4144-aaa2-51130c04d41a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-db96da0b-54f0-4d2e-ac14-14fa952fd05d' class='xr-var-data-in' type='checkbox'><label for='data-db96da0b-54f0-4d2e-ac14-14fa952fd05d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.       , 1.028401 , 1.041853 , ..., 1.0574523, 1.0579894,
        1.0579759],
       [1.       , 1.028401 , 1.041853 , ..., 1.0574523, 1.0579894,
        1.0579759],
       [1.       , 1.0284263, 1.0419334, ..., 1.0625523, 1.0632463,
        1.063455 ],
       ...,
       [1.       , 1.0311841, 1.0512292, ..., 1.1991189, 1.2003735,
        1.1991463],
       [1.       , 1.0311841, 1.0512292, ..., 1.1991189, 1.2003735,
        1.1991463],
       [1.       , 1.0315639, 1.0525478, ..., 1.1986139, 1.2003256,
        1.1991854]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>P</span></div><div class='xr-var-dims'>(id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.0 0.001166 ... 0.1966 0.1972</div><input id='attrs-42db13b3-92db-44af-8a71-050ec16c49b8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-42db13b3-92db-44af-8a71-050ec16c49b8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c530276b-4c11-47ad-ab08-3b269ff096a5' class='xr-var-data-in' type='checkbox'><label for='data-c530276b-4c11-47ad-ab08-3b269ff096a5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.        , 0.00116552, 0.00368537, ..., 0.05631262, 0.05649672,
        0.0567293 ],
       [0.        , 0.00116552, 0.00368537, ..., 0.05631262, 0.05649672,
        0.0567293 ],
       [0.        , 0.00116617, 0.00368967, ..., 0.06025178, 0.06061383,
        0.06101767],
       ...,
       [0.        , 0.00123714, 0.0041773 , ..., 0.19580685, 0.19626497,
        0.1969157 ],
       [0.        , 0.00123714, 0.0041773 , ..., 0.19580685, 0.19626497,
        0.1969157 ],
       [0.        , 0.00124681, 0.0042454 , ..., 0.19619443, 0.1965557 ,
        0.19715704]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>H</span></div><div class='xr-var-dims'>(id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.0 0.0004788 ... 0.3338 0.3459</div><input id='attrs-135dd171-4213-4120-b07f-a5b81847accb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-135dd171-4213-4120-b07f-a5b81847accb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1a51a5a8-e9a2-4d80-be0c-273bf06fba75' class='xr-var-data-in' type='checkbox'><label for='data-1a51a5a8-e9a2-4d80-be0c-273bf06fba75' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.        , 0.00047879, 0.00155762, ..., 0.0982646 , 0.10172348,
        0.10521246],
       [0.        , 0.00047879, 0.00155762, ..., 0.0982646 , 0.10172348,
        0.10521246],
       [0.        , 0.00047905, 0.00155941, ..., 0.10270751, 0.10647892,
        0.11028886],
       ...,
       [0.        , 0.00050799, 0.00176223, ..., 0.31790355, 0.3298536 ,
        0.3419028 ],
       [0.        , 0.00050799, 0.00176223, ..., 0.31790355, 0.3298536 ,
        0.3419028 ],
       [0.        , 0.00051194, 0.00179055, ..., 0.32189065, 0.33382246,
        0.34586832]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>survival</span></div><div class='xr-var-dims'>(id, time)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>1.0 0.9995 0.9984 ... 0.7162 0.7076</div><input id='attrs-d942865f-f85a-41db-8560-0de065cab9d2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d942865f-f85a-41db-8560-0de065cab9d2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3be00e4a-f2eb-4cd6-92cc-4b36e53b34bf' class='xr-var-data-in' type='checkbox'><label for='data-3be00e4a-f2eb-4cd6-92cc-4b36e53b34bf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.        , 0.9995213 , 0.9984436 , ..., 0.906409  , 0.9032793 ,
        0.90013325],
       [1.        , 0.9995213 , 0.9984436 , ..., 0.906409  , 0.9032793 ,
        0.90013325],
       [1.        , 0.9995211 , 0.9984418 , ..., 0.9023909 , 0.89899397,
        0.8955754 ],
       ...,
       [1.        , 0.9994921 , 0.99823934, ..., 0.72767293, 0.71902895,
        0.7104173 ],
       [1.        , 0.9994921 , 0.99823934, ..., 0.72767293, 0.71902895,
        0.7104173 ],
       [1.        , 0.9994882 , 0.998211  , ..., 0.72477746, 0.7161809 ,
        0.70760566]], dtype=float32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1f1462bf-2ff8-4650-a0d1-d768859a0447' class='xr-section-summary-in' type='checkbox'  ><label for='section-1f1462bf-2ff8-4650-a0d1-d768859a0447' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>id</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-e22061da-e454-4943-8143-6921ca25ccee' class='xr-index-data-in' type='checkbox'/><label for='index-e22061da-e454-4943-8143-6921ca25ccee' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([&#x27;101_0&#x27;, &#x27;101_1&#x27;, &#x27;106_0&#x27;, &#x27;106_1&#x27;, &#x27;112_0&#x27;, &#x27;112_1&#x27;, &#x27;118_0&#x27;, &#x27;118_1&#x27;,
       &#x27;124_0&#x27;, &#x27;124_1&#x27;,
       ...
       &#x27;65_3&#x27;, &#x27;65_4&#x27;, &#x27;65_5&#x27;, &#x27;66_0&#x27;, &#x27;66_1&#x27;, &#x27;66_2&#x27;, &#x27;66_3&#x27;, &#x27;66_4&#x27;, &#x27;66_5&#x27;,
       &#x27;6_0&#x27;],
      dtype=&#x27;object&#x27;, name=&#x27;id&#x27;, length=202))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>time</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-afd95d38-dc08-4574-92d4-43badc03b1a2' class='xr-index-data-in' type='checkbox'/><label for='index-afd95d38-dc08-4574-92d4-43badc03b1a2' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([ 24.0,  25.5,  27.0,  30.0,  32.0,  33.0,  34.0,  36.0,  48.0,  54.0,
        60.0,  72.0,  74.0,  75.0,  78.0,  81.0,  84.0,  96.0, 104.0, 108.0,
       114.0, 117.0, 120.0],
      dtype=&#x27;float64&#x27;, name=&#x27;time&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b8653516-2489-4781-bd31-dfc0766c30d1' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b8653516-2489-4781-bd31-dfc0766c30d1' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



By using JAX, the 202 ODE systems needed to integrate all datasets into one model could be evaluated very efficiently resulting in a model evaluation time of 10 ms for 1 iteration after compilation.


```python
sim.benchmark(n=100)
```

    
    Benchmarking with 100 evaluations
    =================================
    Starting Benchmark(time=2025-03-01 18:02:22, )
    Finished Benchmark(runtime=1.2536749839782715s, cputime=1.2531372069999804s, ncores=4
    =================================
    


## Numpyro framwork for bayesian parameter inference

Because diffrax solvers provide gradients of the solutions of an ODE system with respect to its parameters it is possible to use gradient based solvers in conjuction with the ODE solvers. This makes enables us to use gradient based bayesian estimation techniques to assess the uncertainty of the parameters. The most prominent gradient based solver is the No-U-Turn-Sampler (NUTS) by Hofman and Gelman [Hoffman.2011]. It is implemented in the inference framework `numpyro` that is used for this case study.  


```python
# set up the inferer properly
sim.coordinates["time"] = sim.observations.time.values
sim.dispatch_constructor()
sim.set_inferer("numpyro")
```

    Jax 64 bit mode: False
    Absolute tolerance: 1e-06



First of all prior predictions are generated. These are helpful to diagnose
the model and also to compare posterior parameter estimates with the prior
distributions. If there is a large bias, this information can help to achieve
a better model fit.


```python
# set the inference model
seed = 1
prior_predictions = sim.inferer.prior_predictions(n=100, seed=seed)
```

### Problems of gradient based samplers for complex models and large amounts of data

Still a computational problem remains, because for using NUTS, the likelihood function (and its gradients) need to be computed for each data point.
In the given dataset, this means 1426 gradient evaluations with respect to all model parameters per leapfrog step (the number of leapfrosteps varied between 1--1023 per iteration).
This easily scales to dimensions where gradient based MCMC approaches, like NUTS have difficulties, especially when the ODE model and therefore the likelihood function and its gradients, becomes more complex.
For simple problem like the 4-parameter GUTS model $k_d$, $k_k$, $h_b$, $z$, solving the problem with a NUTS approach is feasible (walltime $\approx 30$ minutes), but with more complex models with higher number of parameters, NUTS approaches quickly becomes infeasible (walltime > 48 h). 
In these situation, posteriors were approximated with stochastic variational inference (SVI) [Blei.2017], which estimates posterior distributions, based on finding a parametric distribution that approximates the true, unknown posterior distribution.
While these methods, are constrained to deliver parametric posteriors, they were in good agreement with the posteriors produced by the NUTS algorithm.

### Estimating the parameters with MAP and SVI

In the next step, we take the full model, including deterministic ODE solution and error model and run our maximum-a-posteriori (MAP) estimator on it, with the parameters that have been setup before. The MAP estimator converges of the modes of the parameter distributions (so the most likely value) and *only* differs from maximum likelihood methods in that way that it also accounts for the assumed prior distributions. Note that if the priors were unconstrained uniform the method would be equivalent to the maximum likelihood method (and be only guided by the data).

Because of the speed of the diffrax solver, the model can be fitted in reasonable time (< 5 minutes)

#### Using MAP

| 🛑 | Are you getting a `Permission denied` error when executing the next cell? This is caused by locked results files by `datalad`. Follow the installation instructions in the README 📝. The clue is to unlock 🔓 the results folder: `datalad unlock case_studies/tktd_rna_pulse/results` |
|----|---|


```python
# set the inference model
sim.config.inference_numpyro.kernel = "map"
sim.config.inference_numpyro.svi_iterations = 500
sim.config.inference_numpyro.svi_learning_rate = 0.01
sim.dispatch_constructor(throw_exception=False)
sim.inferer.run()
```

                      Trace Shapes:         
                       Param Sites:         
                      Sample Sites:         
     k_i_substance_normal_base dist      3 |
                              value      3 |
    r_rt_substance_normal_base dist      3 |
                              value      3 |
    r_rd_substance_normal_base dist      3 |
                              value      3 |
    v_rt_substance_normal_base dist      3 |
                              value      3 |
    z_ci_substance_normal_base dist      3 |
                              value      3 |
     k_p_substance_normal_base dist      3 |
                              value      3 |
     k_m_substance_normal_base dist      3 |
                              value      3 |
     h_b_substance_normal_base dist      3 |
                              value      3 |
       z_substance_normal_base dist      3 |
                              value      3 |
      kk_substance_normal_base dist      3 |
                              value      3 |
        sigma_cint_normal_base dist        |
                              value        |
        sigma_nrf2_normal_base dist        |
                              value        |
                      cint_obs dist 202 23 |
                              value 202 23 |
                      nrf2_obs dist 202 23 |
                              value 202 23 |
                  survival_obs dist 202 23 |
                              value 202 23 |


    100%|██████████| 500/500 [00:16<00:00, 30.41it/s, init loss: 6928.1533, avg. loss [476-500]: 622.9951] 
    arviz - WARNING - Shape validation failed: input_shape: (1, 1), minimum_shape: (chains=1, draws=4)


                                    mean  sd    hdi_3%   hdi_97%  mcse_mean  \
    ci_max[101_0]               1757.000 NaN  1757.000  1757.000        NaN   
    ci_max[101_1]               1757.000 NaN  1757.000  1757.000        NaN   
    ci_max[106_0]               1757.000 NaN  1757.000  1757.000        NaN   
    ci_max[106_1]               1757.000 NaN  1757.000  1757.000        NaN   
    ci_max[112_0]               1757.000 NaN  1757.000  1757.000        NaN   
    ...                              ...  ..       ...       ...        ...   
    z_ci_substance[diclofenac]     1.383 NaN     1.383     1.383        NaN   
    z_ci_substance[naproxen]       1.950 NaN     1.950     1.950        NaN   
    z_substance[diuron]            1.500 NaN     1.500     1.500        NaN   
    z_substance[diclofenac]        2.109 NaN     2.109     2.109        NaN   
    z_substance[naproxen]          2.678 NaN     2.678     2.678        NaN   
    
                                mcse_sd  ess_bulk  ess_tail  r_hat  
    ci_max[101_0]                   NaN       NaN       NaN    NaN  
    ci_max[101_1]                   NaN       NaN       NaN    NaN  
    ci_max[106_0]                   NaN       NaN       NaN    NaN  
    ci_max[106_1]                   NaN       NaN       NaN    NaN  
    ci_max[112_0]                   NaN       NaN       NaN    NaN  
    ...                             ...       ...       ...    ...  
    z_ci_substance[diclofenac]      NaN       NaN       NaN    NaN  
    z_ci_substance[naproxen]        NaN       NaN       NaN    NaN  
    z_substance[diuron]             NaN       NaN       NaN    NaN  
    z_substance[diclofenac]         NaN       NaN       NaN    NaN  
    z_substance[naproxen]           NaN       NaN       NaN    NaN  
    
    [2257 rows x 9 columns]



    
![png](tktd_rna_3_6c_substance_specific_files/tktd_rna_3_6c_substance_specific_15_3.png)
    



```python
# show (and explore idata)
print(sim.inferer.idata)
```

    Inference data with groups:
    	> posterior
    	> posterior_predictive
    	> log_likelihood
    	> observed_data
    	> unconstrained_posterior
    	> posterior_model_fits
    	> posterior_residuals
    	> posterior
    	> posterior_predictive
    	> log_likelihood
    	> observed_data
    	> posterior_model_fits
    	> posterior_residuals


We see that the loss curve has quickly converged on the best value, so with the learning rate, we applied, we could probably get the correct inference with fewer iterations. Using the MAP estimator is an excellent way to do model development in a bayesian setting. It gets rid of long parameter estimation runtimes and incorporates prior distributions in the fitting procedure.

#### Posterior predictions

In order to evaluate the goodness of fit for the posteriors, we are looking
at the posterior predictions.

In order to obtain smoother trajectories, the time resolution is increased,
and posterior predictions are calculated.


```python
sim.coordinates["time"] = np.linspace(24, 120, 100)
sim.config.inference.n_predictions = 1
seed = int(np.random.random_integers(0, 100, 1))

sim.dispatch_constructor()
res = sim.inferer.posterior_predictions(n=1, seed=seed).mean(("draw", "chain"))
print(res)
```

    Posterior predictions:   0%|          | 0/1 [00:00<?, ?it/s]

    Posterior predictions: 100%|██████████| 1/1 [00:02<00:00,  2.41s/it]

    <xarray.Dataset>
    Dimensions:          (id: 202, time: 100)
    Coordinates:
      * id               (id) object '101_0' '101_1' '106_0' ... '66_4' '66_5' '6_0'
      * time             (time) float64 24.0 24.97 25.94 26.91 ... 118.1 119.0 120.0
        hpf              (id) float64 24.0 24.0 24.0 24.0 ... 24.0 24.0 24.0 24.0
        nzfe             (id) float64 nan nan nan nan nan ... 9.0 9.0 9.0 9.0 20.0
        treatment_id     (id) int64 101 101 106 106 112 112 118 ... 66 66 66 66 66 6
        experiment_id    (id) int64 36 36 36 36 36 36 36 36 ... 27 27 27 27 27 27 1
        substance        (id) <U10 'diuron' 'diuron' ... 'naproxen' 'naproxen'
        substance_index  (id) int64 0 0 0 0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2 2 2 2
    Data variables:
        cext             (id, time) float32 2.34 2.34 2.34 ... 349.5 349.5 349.5
        cint             (id, time) float32 0.0 12.11 24.19 ... 2.481e+03 2.434e+03
        nrf2             (id, time) float32 1.0 1.075 1.118 ... 3.414 3.387 3.359
        P                (id, time) float32 0.0 0.0006772 0.00229 ... 1.506 1.518
        H                (id, time) float32 0.0 1.628e-06 3.256e-06 ... 1.258 1.277
        survival         (id, time) float32 1.0 1.0 1.0 1.0 ... 0.2898 0.2841 0.2788


    


Next, we plot the predictions against some selected experiments, in order to not be overwhelmed by the data. Note that the observations,
may be slightly diverging from the MAP predictions, because
+ the model is not completely correct
+ other (not displayed) data *pull* the posterior estimate away from the displayed data.

You can select different experiments in the `experiment_selection_1.json` file (or provide a different file).


```python
with open(sim.scenario_path + "/experiment_selection_1.json", "r") as fp:
    data_structure = json.load(fp)
    
res = res.assign_coords({"substance": sim.observations.substance})
cmap = mpl.colormaps["cool"]
fig, axes = plt.subplots(len(data_structure), 3, sharex=True, figsize=(15,10))


for r, (v, vdict) in enumerate(data_structure.items()):
    for c, (s, sdict) in enumerate(vdict["substances"].items()):
        sdata = sim.observations.where(sim.observations.substance == s, drop=True)
        C = np.round(sdata.cext_nom.values, 1)
        norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())
        for eid in sdict["experiment_ids"]:

            ax, meta, obs_ids, _ = sim._plot.plot_experiment(
                self=sim,
                experiment_id=eid,
                substance=s,
                data_var=v,
                cmap=cmap,
                norm=norm,
                ax=axes[r, c]
            )

            if v != "survival":
                ax.set_xlabel("")

            if v == "P":
                ax.set_ylabel("Protein")
                ax.spines[["right", "top"]].set_visible(False)

            if v == "nrf2":
                ax.set_ylim(0, 4)
                # note that the thresholds are mixed up. Diuron and Diclofenac should swap
                z = sim.inferer.idata.posterior.z.mean(("chain", "draw")).values
                ax.hlines(z[c], -10, 120, color="black", lw=0.5)

            if c != 0:
                ax.set_ylabel("")

            l = ax.get_legend()
            if l is not None:
                l.remove()
            if v == "cext":
                ax.set_title(s.capitalize())
            else:
                ax.set_title("")

            res_ids = sim.get_ids(res, {"substance": s, "experiment_id": eid})

            for i in res_ids:
                y = res.sel(id=i)
                ax.plot(res.time, y[v], color=cmap(norm(y.cext.isel(time=0))))

```


    
![png](tktd_rna_3_6c_substance_specific_files/tktd_rna_3_6c_substance_specific_21_0.png)
    


In this figure, we can perfectly see, what is still missing in the model. 

+ Diuron can only model the decay dynamic with an initial overshoot of internal concentrations
+ Multiphase internal concentration dynamics (Diclofenac, Naproxen) cannot be modelled with the simple assumptions taken
+ Secondary pulse in Diuron *nrf2* expression cannot be modeled with the current assumptions. Two major hypotheses compete for explanation: 1. Secondary metabolites re-active *nrf2*. 2. Early stage activation of nascent *nrf2* transcripts bound to KEAP1 proteins explain the first pulse, followed by a second pulse from sustained chemical concentration in the cytoplasm.

Nevertheless, note that the experiments, integrated here into a single model originate from 7 years of experimental work. The resulting experimental uncertainty is bound to be significant, explaining the large variation in observations. 

#### Assessing uncertainty using SVI

SVI converges on a posterio much faster than NUTS and is suited to analyse the uncertainty of the posterior.


```python
# set the inference model
sim.config.inference_numpyro.kernel = "svi"
sim.config.inference_numpyro.svi_iterations = 5_000
sim.config.inference_numpyro.svi_learning_rate = 0.005
sim.coordinates["time"] = sim.observations.time.values
sim.dispatch_constructor(throw_exception=False)
sim.inferer.run()
```

                      Trace Shapes:         
                       Param Sites:         
                      Sample Sites:         
     k_i_substance_normal_base dist      3 |
                              value      3 |
    r_rt_substance_normal_base dist      3 |
                              value      3 |
    r_rd_substance_normal_base dist      3 |
                              value      3 |
    v_rt_substance_normal_base dist      3 |
                              value      3 |
    z_ci_substance_normal_base dist      3 |
                              value      3 |
     k_p_substance_normal_base dist      3 |
                              value      3 |
     k_m_substance_normal_base dist      3 |
                              value      3 |
     h_b_substance_normal_base dist      3 |
                              value      3 |
       z_substance_normal_base dist      3 |
                              value      3 |
      kk_substance_normal_base dist      3 |
                              value      3 |
        sigma_cint_normal_base dist        |
                              value        |
        sigma_nrf2_normal_base dist        |
                              value        |
                      cint_obs dist 202 23 |
                              value 202 23 |
                      nrf2_obs dist 202 23 |
                              value 202 23 |
                  survival_obs dist 202 23 |
                              value 202 23 |


    100%|██████████| 5000/5000 [01:26<00:00, 57.71it/s, init loss: 6351.0991, avg. loss [4751-5000]: 712.7646]    
    arviz - WARNING - Shape validation failed: input_shape: (1, 1000), minimum_shape: (chains=2, draws=4)


                                    mean     sd    hdi_3%   hdi_97%  mcse_mean  \
    ci_max[101_0]               1757.000  0.000  1757.000  1757.000      0.000   
    ci_max[101_1]               1757.000  0.000  1757.000  1757.000      0.000   
    ci_max[106_0]               1757.000  0.000  1757.000  1757.000      0.000   
    ci_max[106_1]               1757.000  0.000  1757.000  1757.000      0.000   
    ci_max[112_0]               1757.000  0.000  1757.000  1757.000      0.000   
    ...                              ...    ...       ...       ...        ...   
    z_ci_substance[diclofenac]     0.969  0.080     0.827     1.125      0.003   
    z_ci_substance[naproxen]       1.704  0.082     1.559     1.871      0.003   
    z_substance[diuron]            1.474  0.089     1.306     1.645      0.003   
    z_substance[diclofenac]        2.248  0.072     2.116     2.386      0.002   
    z_substance[naproxen]          2.767  0.147     2.506     3.054      0.005   
    
                                mcse_sd  ess_bulk  ess_tail  r_hat  
    ci_max[101_0]                 0.000    1000.0    1000.0    NaN  
    ci_max[101_1]                 0.000    1000.0    1000.0    NaN  
    ci_max[106_0]                 0.000    1000.0    1000.0    NaN  
    ci_max[106_1]                 0.000    1000.0    1000.0    NaN  
    ci_max[112_0]                 0.000    1000.0    1000.0    NaN  
    ...                             ...       ...       ...    ...  
    z_ci_substance[diclofenac]    0.002     819.0     820.0    NaN  
    z_ci_substance[naproxen]      0.002     967.0    1024.0    NaN  
    z_substance[diuron]           0.002    1052.0     944.0    NaN  
    z_substance[diclofenac]       0.002     986.0     948.0    NaN  
    z_substance[naproxen]         0.003     933.0     935.0    NaN  
    
    [2257 rows x 9 columns]



    
![png](tktd_rna_3_6c_substance_specific_files/tktd_rna_3_6c_substance_specific_24_3.png)
    



```python
sim.inferer.store_results(f"{sim.output_path}/numpyro_svi_posterior.nc")
```

#### Posterior predictions with uncertainty intervals of a single estimate

Although the estimates under parameter identifiability issues are not reliable, we can still plot the results


```python
sim.config.inference.n_predictions = 100
sim.coordinates["time"] = np.linspace(24, 120, 200)
sim.seed=1
sim.config.data_structure.remove("lethality")
sim.dispatch_constructor()
_ = sim._plot.pretty_posterior_plot_multisubstance(sim, save=False, show=True)
```

    Deleted 'lethality' DataVariable(dimensions=['id', 'time'] min=0.0 max=18.0 observed=False dimensions_evaluator=None).
    PRETTY PLOT: starting...


    Posterior predictions: 100%|██████████| 100/100 [00:05<00:00, 19.23it/s]


    PRETTY PLOT: make predictions for Diuron in bin (1/5)
    PRETTY PLOT: make predictions for Diuron in bin (2/5)
    PRETTY PLOT: make predictions for Diuron in bin (3/5)
    PRETTY PLOT: make predictions for Diuron in bin (4/5)
    PRETTY PLOT: make predictions for Diuron in bin (5/5)



    
![png](tktd_rna_3_6c_substance_specific_files/tktd_rna_3_6c_substance_specific_27_3.png)
    


    PRETTY PLOT: make predictions for Diclofenac in bin (1/4)
    PRETTY PLOT: make predictions for Diclofenac in bin (2/4)
    PRETTY PLOT: make predictions for Diclofenac in bin (3/4)
    PRETTY PLOT: make predictions for Diclofenac in bin (4/4)



    
![png](tktd_rna_3_6c_substance_specific_files/tktd_rna_3_6c_substance_specific_27_5.png)
    


    PRETTY PLOT: make predictions for Naproxen in bin (1/6)
    PRETTY PLOT: make predictions for Naproxen in bin (2/6)
    PRETTY PLOT: make predictions for Naproxen in bin (3/6)
    PRETTY PLOT: make predictions for Naproxen in bin (4/6)
    PRETTY PLOT: make predictions for Naproxen in bin (5/6)
    PRETTY PLOT: make predictions for Naproxen in bin (6/6)



    
![png](tktd_rna_3_6c_substance_specific_files/tktd_rna_3_6c_substance_specific_27_7.png)
    


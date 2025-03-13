# Installation

To install pymob, run `pip install pymob`

Backends can be installed with e.g. `pip install pymob[numpyro]`.
Other available backends are listed in {ref}`inference-backends`

## Development

If you want to develop `pymob`, clone the repository and install pymob as an editable package

```bash
git clone git@github.com:flo-schu/pymob.git
cd pymob
conda create -n pymob python=3.11
pip install -e .[dev,docs]
pre-commit install
```

Further inference backends may be installed with `pip install -e .[numpyro,pyabc,pymoo,interactive]`

## Pre-release versions

Pymob is under active development. It is used and developed within multiple projects simultaneously and in order to maintain a consistent release history, the main work is done in project-branches which contain the most cutting-edge features. These can always checked out locally, but may not be working correctly. Instead, it is recommended to install or checkout alpha-versions. 

E.g. `pip install pymob==0.3.0a5` which is the 5th alpha release of a project branch that was based on pymob v0.2.x.

```{warning}
It may be possible that different projects release on the same minor version. In this case the release notes (https://github.com/flo-schu/pymob/releases) should be reviewed to see which project it refers to.
```

#!/bin/bash

EXECUTE_NOTEBOOKS=$1

# Check the value of the environmental variable
if [ "$EXECUTE_NOTEBOOKS" == "true" ]; then
    nb_exec="--execute"
else
    nb_exec=""
fi

jupyter nbconvert --to markdown ${nb_exec} case_studies/lotka_volterra_case_study/scripts/*.ipynb --output-dir=docs/source/examples/lotka_volterra_case_study/
jupyter nbconvert --to markdown ${nb_exec} case_studies/tktd_rna_pulse/scripts/*.ipynb --output-dir=docs/source/examples/tktd_rna_pulse/
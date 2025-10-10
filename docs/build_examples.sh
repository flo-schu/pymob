#!/bin/usr/bash


EXECUTE_NOTEBOOKS=$1

# Check the value of the environmental variable
if [ "$EXECUTE_NOTEBOOKS" == "no-execute" ]; then
    nb_exec=""
else
    nb_exec="--execute"
fi


update_repo() {
  local REPO=$1
  local DIRECTORY=$2
  local CWD=$PWD

  if [ ! -d "$DIRECTORY" ]; then
    # clone if it does not exist
    git clone "$REPO" "$DIRECTORY"
  else
    # update if it exists
    cd $DIRECTORY
    git pull
    cd $CWD
  fi
}


CASE_STUDY_DIR="./docs/source/examples/case_studies"

# lotka volterra
REPO="https://github.com/flo-schu/lotka_volterra_case_study.git"
DIRECTORY="lotka_volterra_case_study"
update_repo $REPO $DIRECTORY

jupyter nbconvert --to markdown ${nb_exec} "$CASE_STUDY_DIR/$DIRECTORY/scripts/hierarchical_model.ipynb" --output-dir="docs/source/examples/$(basename "$DIRECTORY")/"
jupyter nbconvert --to markdown ${nb_exec} "$CASE_STUDY_DIR/$DIRECTORY/scripts/hierarchical_model_varying_y0.ipynb" --output-dir="docs/source/examples/$(basename "$DIRECTORY")/"


# Tktd rna pulse
REPO="https://github.com/flo-schu/tktd_rna_pulse.git"
DIRECTORY="tktd_rna_pulse"
echo "$PWD"
update_repo $REPO $DIRECTORY

jupyter nbconvert --to markdown ${nb_exec} "$CASE_STUDY_DIR/$DIRECTORY/scripts/tktd_rna_5_*.ipynb" --output-dir="docs/source/examples/$(basename "$DIRECTORY")/"
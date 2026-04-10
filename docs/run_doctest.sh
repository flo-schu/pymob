#!/bin/bash
cd pymob
pytest --doctest-modules --disable-warnings \
    --ignore=inference/interactive.py \
    --ignore=inference/sbi \
    --ignore=inference/optimization.py

rm -rf case_studies/testing
rmdir case_studies
cd ..

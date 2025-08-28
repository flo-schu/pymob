#!/usr/bin/env bash

source activate pymob
pytest -m "not slow"
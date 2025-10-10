#!/bin/usr/bash
sphinx-apidoc  -o docs/source/api pymob && sphinx-build -M html docs/source/ docs/build/

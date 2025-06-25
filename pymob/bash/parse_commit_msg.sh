#!/bin/bash

# Input string
input_string="$1"

# Regular expression to match the required pattern
# linux (for local reference)
# regex="^bump version [0-9]+\.[0-9]+\.[0-9]+?[a-zA-Z]+?[0-9]* -> [0-9]+\.[0-9]+\.[0-9]+?[a-zA-Z]+?[0-9]*$"
# github 
regex="^bump\ version\ [0-9]+\.[0-9]+\.[0-9]+?[a-zA-Z]+?[0-9]+\ -\>\ [0-9]+\.[0-9]+\.[0-9]+?[a-zA-Z]+?[0-9]+$"

if [[ $input_string =~ $regex ]]; then
  echo true
else
  echo false
fi

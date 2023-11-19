# created by Florian Schunck on 01.07.2020
# Project: timepath
# Short description of the feature:
# - add round to nearest integer of X
# 
# ------------------------------------------------------------------------------
# Open tasks:
#
# ------------------------------------------------------------------------------
import numpy as np

def round_to_nearest_multiple_of_y(x, y):
    return np.round(x / y) * y

def take_mean(list):
    if len(list) > 0:
        return np.mean(list)
    else:
        return np.nan

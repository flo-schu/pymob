# created by Florian Schunck on 01.07.2020
# Project: timepath
# Short description of the feature:
# - miscellaneous helper functions
#
# ------------------------------------------------------------------------------
# Open tasks:
#
# ------------------------------------------------------------------------------
from typing import List
from datetime import timedelta
import itertools as it
import socket
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import ticker, dates


def repeat_func(func, n, **kwargs):
    """
    repeat a generic function and store the output in a numpy array
    func: function to be called
    n: number of repetitions of the function call
    **kwargs: keyword arguments passed to func

    returns a 1D numpy array of the iterated results
    """
    result = np.zeros(n).tolist()
    for i in range(n):
        result[i] = func(**kwargs)
    return result


def invert_list_tuple(data):
    """
    this small helper inverts a list of tuples to a list of lists like
    [(a,b),(a,b),(a,b)] --> [[a,a,a],[b,b,b]]
    """
    # obtain number of results in tuple
    n_results = len(data[0])
    n_data = len(data)

    results = np.ndarray((n_results, n_data)).tolist()

    for i in range(n_data):
        for k in range(n_results):
            results[k][i] = data[i][k]

    return results


def pop_key(d, ex):
    """
    pop (remove) a given key from a dictionary
    """
    for i in ex:
        d.pop(i)

    return d


def round_decimals_down(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.floor(number)

    factor = 10 ** decimals
    return np.floor(number * factor) / factor

def expand_grid(data_dict):
    """Create a dataframe from every combination of given values."""
    rows = it.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def get_limits_from_array(theta):
    limits = []
    spread = np.ptp(theta, axis=0).tolist()
    means = theta.mean(axis=0).tolist()
    for i in range(theta.shape[1]):
        lim = [(means[i] - spread[i]/2),
               (means[i] + spread[i]/2)]
        limits.append(lim)

    return limits

def match_columns(features, dataset):
    # match sim and experimental data columns
    exp_cols = [True if l in features else False for l in dataset[
        "labels"]["values"]]
    return exp_cols


def dayTicks(x, pos):
    x = timedelta(seconds=x / 10**9)  # format nanoseconds to seconds
    return str(x.days)

def hourTicks(x, pos):
    x = timedelta(seconds=x / 10**9)  # format nanoseconds to seconds
    return str(int(x.total_seconds() / 3600))

class Date2Delta: 
    def __init__(self, origin):
        self.origin = dates.date2num(origin)
    
    def __call__(self, x, pos):
        delta = x - self.origin
        return str(int(delta))


def get_host(localhosts):
    if socket.gethostname() in localhosts:
        return "local"
    
    return "remote"


def replace_na(df, column, default_value):
    df[column] = df[column].fillna(default_value)
    return df

def get_grouped_unique_val(df, variable, groupby="id", extra_dim=None, fill_extra=np.nan):
    unqiue_value_per_level = []
    for key, group in df.groupby(groupby):
        values = group[variable].dropna().unique()
        if extra_dim is not None:
            coords = df[extra_dim].unique()
            found_coords = group[extra_dim].dropna().unique()

            values_extradim = np.repeat(fill_extra, len(coords))
            for val, co in zip(values, found_coords):
                values_extradim[co == coords] = val              

            unqiue_value_per_level.append(values_extradim)

        else:
            if len(values) > 1:
                raise ValueError(f"{key} has more than one unique values {values}")
            elif len(values) == 0:
                values = [np.nan]
        
            unqiue_value_per_level.append(values[0])

    return np.array(unqiue_value_per_level)


def pivot_xarray(ds, name, dimname, sep="_"):
    variable_names = list(ds.keys())
    old_vars = [var for var in variable_names if name + sep in var]
    ds_sub = ds[old_vars]
    da = ds_sub.to_array(dim=dimname, name=name)
    coords = [val.replace(name + sep, "") for val in da[dimname].values]
    da = da.assign_coords({dimname:coords})
    ds = ds.drop(old_vars)
    ds[name] = da

    return ds


def to_xarray_dataset(rawdata):
    data = xr.DataArray(rawdata["data"], coords=rawdata["labels"], name="a")
    dataset = data.to_dataset(dim="values")
    dataset = dataset.assign_attrs(**rawdata["meta"])
    return dataset



DAYFMT = ticker.FuncFormatter(dayTicks)
HOURFMT = ticker.FuncFormatter(hourTicks)

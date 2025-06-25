import warnings
from typing import Literal, Union, List, Optional, Dict
from functools import partial

import numpy as np
import xarray as xr
import arviz as az
import pandas as pd
from arviz.sel_utils import xarray_var_iter
from matplotlib import pyplot as plt
import matplotlib as mpl

from pymob.utils.plot_helpers import plot_hist, plot_loghist

def cluster_chains(posterior, deviation="std"):
    assert isinstance(posterior, (xr.DataArray, xr.Dataset))
    chain_means = posterior.mean(dim="draw")
    if deviation == "std":
        chain_dev = posterior.std(dim="draw")
    elif "frac:" in deviation:
        _, frac = deviation.split(":")
        chain_dev = chain_means * float(frac)
    else:
        raise ValueError("Deviation method not implemented.")    

    global cluster_id
    cluster_id = 0
    cluster = [cluster_id] * len(posterior.chain)
    unclustered_chains = posterior.chain.values

    def recurse_clusters(unclustered_chains):
        global cluster_id
        compare = unclustered_chains[0]
        new_cluster = []
        for i in unclustered_chains[1:]:
            a = chain_means.sel(chain=compare) + chain_dev.sel(chain=compare) > chain_means.sel(chain=i)
            b = chain_means.sel(chain=compare) - chain_dev.sel(chain=compare) < chain_means.sel(chain=i)
            isin_dev = (a * b).all()

            if isinstance(isin_dev, xr.Dataset):
                isin_dev = isin_dev.to_array().all()

            if not isin_dev:
                cluster[i] = cluster_id + 1
                new_cluster.append(i)

        if len(new_cluster) == 0:
            return

        cluster_id += 1
        recurse_clusters(new_cluster)
    
    recurse_clusters(unclustered_chains)

    if cluster_id > 0:
        warnings.warn(
            "The number of clusters in the InferenceData object was "+
            f"{cluster_id + 1} > 1. This indicates that not all chains/restarts "
            "Converged on the same estimate."
        )

    return cluster


def rename_extra_dims(df, extra_dim_suffix="_dim_0", new_dim="new_dim", new_coords=None):
    # TODO: COuld be used for numypro backend for fixing posterior indexes
    df_ = df.copy()
    data_vars = list(df_.data_vars.keys())

    # swap dimension names for all dims that have the suffix 
    new_dims = {}
    for dv in data_vars:
        old_dim = f"{dv}{extra_dim_suffix}"
        if df_.dims[old_dim] == 1:
            df_[dv] = df_[dv].squeeze(old_dim)
        else:
            new_dims.update({old_dim: new_dim})

    df_ = df_.swap_dims(new_dims)

    # assign coords to new dimension
    df_ = df_.assign_coords({new_dim: new_coords})
    
    # drop renamed coords
    df_ = df_.drop([f"{dv}{extra_dim_suffix}" for dv in data_vars])

    return df_



def plot_trace(idata, var_names, output):
    """
    TODO: Should be outsourced to pymob.sim.plot
    """

    if hasattr(idata, "sample_stats"):
        if "diverging" in idata["sample_stats"]:
            idata["sample_stats"]["diverging"] = idata["sample_stats"].diverging.astype(int)

    if hasattr(idata, "posterior"):
        axes = az.plot_trace(
            idata,
            var_names=var_names
        )
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(output)

    return fig, output

def plot_pairs(idata, var_names, output):
    """
    TODO: Should be outsourced to pymob.sim.plot
    """

    if hasattr(idata, "sample_stats"):
        if "diverging" in idata["sample_stats"]:
            idata["sample_stats"]["diverging"] = idata["sample_stats"].diverging.astype(int)

    if hasattr(idata, "posterior"):
        axes = az.plot_pair(
            idata, 
            divergences=True, 
            var_names=var_names
        )
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(output)

    return fig, output



# plot loghist
def plot_posterior_samples(posterior, col_dim=None, log=True, hist_kwargs={}):
    if log:
        hist = plot_loghist
    else:
        hist = plot_hist

    parameters = list(posterior.data_vars.keys())
    samples = posterior.stack(sample=("chain", "draw"))

    fig = plt.figure(figsize=(5, len(parameters)*2))
    fig.subplots_adjust(right=.95, top=.95, hspace=.25)

    gs = fig.add_gridspec(len(parameters), 1)

    for i, key in enumerate(parameters):
        postpar = samples[key]
        if col_dim in postpar.dims:
            col_coords = postpar[col_dim]
            gs_par = gs[i, 0].subgridspec(1, len(col_coords))
            axes = gs_par.subplots()

            for ax, coord in zip(axes, col_coords):
                hist(
                    x=postpar.sel({col_dim: coord}), 
                    name=f"${key}$ {str(coord.values)}",
                    ax=ax,
                    **hist_kwargs
                )

        else:
            gs_par = gs[i, 0].subgridspec(1, 1)
            ax = gs_par.subplots()

            hist(
                x=postpar, 
                name=f"${key}$",
                ax=ax,
                **hist_kwargs
            )


    return fig


def bic(idata: az.InferenceData):
    """calculate the BIC for az.InferenceData. The function will average over
    all samples from the markov chain
    """
    log_likelihood = idata.log_likelihood.mean(("chain", "draw")).sum().to_array().sum()
    k = idata.posterior.mean(("chain", "draw")).count().to_array().sum()

    vars = [i.split("_")[0] for i in list(idata.log_likelihood.data_vars.keys())]
    n = 0
    for v in vars:
        if v in idata.observed_data:
            n += (~idata.observed_data[v].isnull()).sum()
        elif v + "_obs" in idata.observed_data:
            n += (~idata.observed_data[v + "_obs"].isnull()).sum()
        else:
            raise IndexError(f"Variable {v} or {v+'_obs'} not in idata")

    # n = (~idata.observed_data[vars].isnull()).sum().to_array().sum()

    bic = float(k * np.log(n) - 2 * log_likelihood)
    msg = str(
        "Bayesian Information Criterion (BIC):"
        f"\nParameters: {int(k)}"
        f"\nData points: {int(n)}"
        f"\nLog-likelihood: {float(log_likelihood)}"
        f"\nBIC: {bic}"
    )

    return msg, bic

def log_lik(x):
    dims_sum = [d for d in x.dims if d not in ["chain", "draw"]]
    return x.sum(dims_sum)

def rmse(x_0: xr.DataArray, x: xr.DataArray):
    n = x.count()
    dims_sum = [d for d in x_0.dims if d not in ["chain", "draw"]]
    return np.sqrt((1 / n * np.power(x - x_0, 2).sum(dims_sum)))

def nrmse(x_0: xr.DataArray, x: xr.DataArray, mode: Literal["range", "mean", "iqr"]):
    dims_sum = [d for d in x_0.dims if d not in ["chain", "draw"]]

    if mode == "range":
        return rmse(x_0, x) / (x.max(dims_sum) - x.min(dims_sum))
    elif mode == "mean":
        return rmse(x_0, x) / x.mean(dims_sum)
    elif mode == "mean":
        return rmse(x_0, x) / x.mean(dims_sum)
    else:
        raise NotImplementedError(
            f"Mode {mode} not implemented. Use one of the following modes"
        )

        
    # normalized_obs = idata.observed_data.survival.groupby("id").map(
    #     lambda group: group / group.max(dim="time")
    # )
    # nrmse_draws = 1 / normalized_obs.mean()
    # np.sqrt(1/normalized_obs.count() * ((normalized_obs-idata.posterior_model_fits.survival ** 2).sum(dim=["id","time"]))
    # data_dict["nrmse_mean"].append(100 * float(nrmse_draws.mean().data))
    # data_dict["nrmse_higher"].append(100 * float(az.hdi(nrmse_draws,hdi_prob=0.95).sel(hdi="higher").to_dataarray()[0].data))
    # data_dict["nrmse_lower"].append(100*float(az.hdi(nrmse_draws,hdi_prob=0.95).sel(hdi="lower").to_dataarray()[0].data))

def add_cluster_coordinates(idata: az.InferenceData, deviation="std") -> az.InferenceData:
    """Clusters the chains in the posterior"""
    if "posterior" in idata.groups():
        cluster = cluster_chains(idata.posterior, deviation=deviation)
        idata = idata.assign_coords(cluster=("chain", cluster))
    return idata


def format_parameter(par, subscript_sep="_", superscript_sep="__", textwrap="\\text{}"):
    super_pos = par.find(superscript_sep)
    sub_pos = par.find(subscript_sep)
    
    scripts = sorted(zip([sub_pos, super_pos], [subscript_sep, superscript_sep]))
    scripts = [sep for pos, sep in scripts if pos > -1]


    def wrap_text(substr):

        if len(substr) == 1:
            substring_fmt = f"{substr}"
        else:
            substr = substr.replace("_", " ")
            substring_fmt = textwrap.replace("{}", "{{{}}}").format(substr)
    
        return f"{{{substring_fmt}}}"

    formatted_string = "$"
    for i, sep in enumerate(scripts):
        substr, par = par.split(sep, 1)
        substring_fmt = wrap_text(substr=substr)

        math_sep = "_" if sep == subscript_sep else "^"

        formatted_string += substring_fmt + math_sep

    formatted_string += wrap_text(par) + "$"

    return formatted_string

def round_to_sigfig(num, sig_fig=3):
    """Rounds a number to a specified number of significant figures."""
    if num == 0:
        return num
    if np.isnan(num):
        return num
    return round(num, sig_fig - int(np.floor(np.log10(abs(num)))) - 1)


def create_table(
    posterior, 
    error_metric: Literal["hdi","sd"] = "hdi", 
    vars: Dict = {}, 
    nesting_dimension: Optional[Union[List,str]] = None,
    fmt: Literal["csv", "tsv", "latex"] = "csv",
    significant_figures: int = 3,
    parameters_as_rows: bool = True,
) -> pd.DataFrame:
    """The function is not ready to deal with any nesting dimensionality
    and currently expects the 2-D case
    """
    tab = az.summary(
        posterior, var_names=list(vars.keys()), 
        fmt="xarray", kind="stats", stat_focus="mean", 
        hdi_prob=0.94
    )

    tab = tab.rename(vars)

    if nesting_dimension is None:
        stack_cols = ("metric",)
    else:
        if isinstance(nesting_dimension, str):
            nesting_dimension = [nesting_dimension]
        stack_cols = (*nesting_dimension, "metric")
        
    stack_cols = [s for s in stack_cols if s in tab.coords]


    tab = tab.apply(np.vectorize(
        partial(round_to_sigfig, sig_fig=significant_figures)
    ))


    if error_metric == "sd":
        arrays = []
        for par in vars.values():
            par_formatted = tab.sel(metric=["mean", "sd"])[par]\
                .astype(str).str\
                .join("metric", sep=" ± ")
            arrays.append(par_formatted)


        table = xr.combine_by_coords(arrays)
        table = table.assign_coords(metric="mean ± std").expand_dims("metric")
        table = table.to_dataframe().T

    elif error_metric == "hdi":
        stacked_tab = tab.sel(metric=["mean", "hdi_3%", "hdi_97%"])\
            .assign_coords(metric=["mean", "hdi 3%", "hdi 97%"])\
            .stack(col=stack_cols)
        table = stacked_tab.to_dataframe().T.drop(list(stack_cols))

    else:
        raise NotImplementedError("Must use one of 'sd' or 'hdi'")


    if fmt == "latex":
        table.columns.names = [c.replace('_',' ') for c in table.columns.names]
        table.index = [format_parameter(i) for i in list(table.index)]
        table = table.rename(
            columns={"hdi 3%": "hdi 3\\%", "hdi 97%": "hdi 97\\%"}
        )
    else: 
        pass

    if parameters_as_rows:
        return table
    else:
        return table.T


def filter_not_converged_chains(idata, deviation=1.05):
    posterior = idata.posterior
    log_likelihood = idata.log_likelihood
    log_likelihood_summed = log_likelihood.to_array("obs")
    log_likelihood_summed = log_likelihood_summed.sum(("time", "id", "obs"))

    # filter non-converged parameter estimates
    likelihood_mask = (
        # compares the mean of the summed log likelihood of a given chain
        log_likelihood_summed.mean("draw") > 
        # to the maximum of all chain means times a factor
        log_likelihood_summed.mean("draw").max() * deviation
    )
    posterior_filtered = posterior.where(likelihood_mask, drop=True)
    log_likelihood_filtered = log_likelihood.where(likelihood_mask, drop=True)

    idata = az.InferenceData(
        posterior=posterior_filtered, 
        log_likelihood=log_likelihood_filtered,
        observed_data=idata.observed_data,
    )

    return idata


def log(msg, out, newlines=1, mode="a"):
    with open(out, mode, encoding="utf-8") as f:
        print(msg, file=f, end="\n")
        for _ in range(newlines):
            print("", file=f, end="\n")


def evaluate_posterior(sim, nesting_dimension, n_samples=10_000, vars_table={}, 
                       seed=1, save=True, show=False):
    """The function is not ready to deal with any nesting dimensionality 
    and currently expects the 2-D case
    """
    rng = np.random.default_rng(seed)
    idata = sim.inferer.idata
    # idata.posterior = idata.posterior.chunk(chunks={"draw":100}).load()
    # idata.log_likelihood = idata.log_likelihood.chunk(chunks={"draw":100})
    n_subsample = min(
        int(n_samples / idata.posterior.sizes["chain"]), 
        idata.posterior.sizes["draw"]
    )

    if n_subsample < 250:
        warnings.warn(
            "The number of samples drawn from each chain for the pairplot "
            f"({n_subsample}) may be too small to be representative. "
            "Consider increasing n_samples."
        )

    subsamples = rng.choice(idata.posterior.draw, n_subsample, replace=False)
    idata.posterior = idata.posterior.sel(draw=subsamples)
    idata.log_likelihood = idata.log_likelihood.sel(draw=subsamples)

    log_likelihood_summed = idata.log_likelihood.to_array("obs")
    log_likelihood_summed = log_likelihood_summed.sum(("time", "id", "obs"))

    print(az.summary(idata.posterior))

    table = create_table(
        posterior=idata.posterior, 
        error_metric="hdi",
        vars=vars_table,
        nesting_dimension=nesting_dimension,
        fmt="latex",
    )
    table.index = [format_parameter(i) for i in list(table.index)]

    # bic 
    msg, _ = bic(idata)
    if save:
        log(table, out=f"{sim.output_path}/parameter_table.tex", mode="w")
        log(msg=msg, out=f"{sim.output_path}/bic.md", mode="w")

    if show:
        print(table)
        print(msg)

    fig_param = plot_posterior_samples(
        idata.posterior, 
        col_dim=nesting_dimension, 
        log=True,
        hist_kwargs = dict(hdi=True, bins=20)
    )
    fig_param.set_size_inches(12, 30)

    if save:
        fig_param.savefig(f"{sim.output_path}/multichain_parameter_estimates.jpg")

    if show:
        plt.show()
    else:
        plt.close()

    def plot_pairs(posterior, likelihood):
        parameters = list(posterior.data_vars.keys())

        N = len(parameters)
        parameters_ = parameters.copy()
        fig = plt.figure(figsize=(3*N, 3*(N+1)))
        gs = fig.add_gridspec(N, N+1, width_ratios=[1]*N+[0.2])
        

        i = 0
        while len(parameters_) > 0:
            par_x = parameters_.pop(0)
            hist_ax = gs[i,i].subgridspec(1, 1).subplots()
            plot_hist(
                posterior[par_x].stack(sample=("chain", "draw")), 
                ax=hist_ax, decorate=False, bins=20
            )
            hist_ax.set_title(par_x)
            for j, par_y in enumerate(parameters_, start=i+1):
                ax = gs[j,i].subgridspec(1, 1).subplots()

                scatter = ax.scatter(
                    posterior[par_x], 
                    posterior[par_y], 
                    c=likelihood, 
                    alpha=0.25,
                    s=10,
                    cmap=mpl.colormaps["plasma_r"]
                )

                if j != len(parameters)-1:
                    ax.set_xticks([])
            
                ax.set_xlabel(par_x)            
                ax.set_ylabel(par_y)

            i += 1

        # ax_colorbar = gs[:,N].subgridspec(1, 1).subplots()
        # fig.colorbar(scatter, cax=ax_colorbar)
        return fig

    for coord in idata.posterior[nesting_dimension].values:
        print("=" * len(coord))
        print(coord.capitalize())
        print("=" * len(coord))
        az.plot_trace(idata.posterior.sel({nesting_dimension:coord}))
        
        if save:
            plt.savefig(f"{sim.output_path}/multichain_pseudo_trace_{coord}.jpg")

        if show:
            plt.show()
        else:
            plt.close()

        fig = plot_pairs(
            posterior=idata.posterior.sel({nesting_dimension:coord}), 
            likelihood=log_likelihood_summed,
        )

        if save:
            fig.savefig(f"{sim.output_path}/multichain_pairs_{coord}.jpg")

        if show:
            plt.show()
        else:
            plt.close()



def plot_pair(posterior, likelihood, parameters, axes=None):

    likelihood_vars = list(likelihood.data_vars.keys())
    N = len(likelihood_vars)

    if axes is None:
        fig, axes = plt.subplots(1, N, figsize=(3+(2*(N-1)), 3), sharey=True, sharex=True)
    
    if isinstance(likelihood, xr.DataArray):
        if likelihood.name is None:
            likelihood.name = "joint_likelihood"
        likelihood = likelihood.to_dataset()
    

    par_a, par_b = parameters

    post_a = posterior[par_a]
    post_b = posterior[par_b]

    # always put the smaller parameter on the x-axis
    if post_a.ndim <= post_b.ndim:
        post_x = post_a
        post_y = post_b
    else:
        post_x = post_b.stack(sample=("chain", "draw"))
        post_y = post_a.stack(sample=("chain", "draw"))

    loglik = likelihood.stack(sample=("chain","draw"))
    extra_dims_y = [d for d in post_y.dims if d not in post_x.dims]
    norm = mpl.colors.Normalize(vmin=(likelihood).to_array().min(), vmax=(likelihood).to_array().max())
    cmap = mpl.colormaps["plasma"]

    for likvar, ax in zip(likelihood_vars, axes):
        ll = loglik[likvar]
        ax.set_title(likvar)
        ax.set_xlabel(post_x.name)
        ax.set_ylabel(post_y.name)
        for dim in extra_dims_y:
            for coord in post_y.coords[dim].values:
                if dim in ll.dims:
                    ll_coord = ll.sel({dim:coord})
                else:
                    ll_coord = ll

                scatter = ax.scatter(
                    post_x, 
                    post_y.sel({dim:coord}), 
                    c=ll_coord, 
                    alpha=0.25,
                    marker=f"${coord}$",
                    s=40,
                    cmap=cmap,
                    norm=norm
                )

                # ax.annotate()
        
    cbar = fig.colorbar(
        mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        # if the same axis is used it is plot next to
        # the plot depending on the location option
        # ax=ax,
        # location="right",
        orientation="vertical",

        # if another axis is used, the fraction should be increased to 1.0
        ax=ax,
        fraction=0.1,
        pad=0.05,
        # aspect=20, # parameter is not so important. It thins the colorbar
        label="negative log-likelihood"
    )
    fig.tight_layout()

    # ax_colorbar = gs[:,N].subgridspec(1, 1).subplots()
    return fig
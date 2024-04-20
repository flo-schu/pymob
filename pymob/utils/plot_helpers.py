import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from pymob.utils.misc import Date2Delta
import arviz as az

def plot_loghist(x, name="", bins=10, ax=None, hdi=False, decorate=True, 
                 color="tab:blue", alpha=1, legend=None, orientation="vertical",
                 **hist_kwargs):
    _, bins = np.histogram(x, bins=bins)
    median = np.median(x)
    if hdi:
        assert hasattr(x, "chain"), "chain must be in coordinates, if hdi=True"
        assert hasattr(x, "draw"), "draw must be in coordinates, if hdi=True"
        lq, uq = az.hdi(x.unstack(), hdi_prob=.95).to_array().isel(variable=0).values
    else:
        lq, uq = np.quantile(x, [0.025, 0.975])
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    if ax is None:
        ax = plt.subplot(111)
    lhist, lbins, _ = ax.hist(x, bins=logbins, color=color, alpha=alpha, label=legend, 
                              orientation=orientation, **hist_kwargs)

    # turn of labels for minot ticks

    if decorate:
        plot_kwargs = dict(rotation=90, va="bottom", ha="center")
        bbox=dict(facecolor='white', alpha=0.8, linewidth=0)
        ax.vlines([lq, median, uq], 0, np.max(lhist)*1.1, linestyle="--", 
                    color = ["grey", "black", "grey"])
        ax.text(lq, np.max(lhist)*0.3, f"{lq:.1e}", color="black", bbox=bbox, **plot_kwargs)
        ax.text(uq, np.max(lhist)*0.3, f"{uq:.1e}", color="black", bbox=bbox, **plot_kwargs)
        ax.text(median, np.max(lhist)*0.3, f"{median:.1e}", bbox=bbox, **plot_kwargs)
        ax.text(0.05, 0.95, name, transform=ax.transAxes, ha="left", va="top")

    if orientation=="vertical":
        ax.set_xscale("log")
        ax.set_ylim(0, np.max(lhist)*1.1)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif orientation=="horizontal":
        ax.set_xlim(0, np.max(lhist)*1.1)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        raise RuntimeError("Choose orientation = 'vertical' or 'horizontal'")
    return ax
    # plt.savefig("work/case_studies/core_daphnia/results/expo_control_3/plots/test.png")

def plot_hist(x, name="", bins=10, ax=None, hdi=False, decorate=True):
    median = np.median(x)
    if hdi:
        assert hasattr(x, "chain"), "chain must be in coordinates, if hdi=True"
        assert hasattr(x, "draw"), "draw must be in coordinates, if hdi=True"
        lq, uq = az.hdi(x.unstack(), hdi_prob=.95).x.values
    else:
        lq, uq = np.quantile(x, [0.025, 0.975])
    if ax is None:
        ax = plt.subplot(111)
    lhist, lbins, _ = ax.hist(x, bins=bins)

    if decorate:
        plot_kwargs = dict(rotation=90, va="bottom", ha="center")
        bbox=dict(facecolor='white', alpha=0.5, edgecolor=None)
        ax.vlines([lq, median, uq], 0, np.max(lhist)*1.1, linestyle="--", 
                    color = ["grey", "black", "grey"])
        ax.text(lq, np.max(lhist)*0.3, f"{lq:.1e}", color="black", bbox=bbox, **plot_kwargs)
        ax.text(uq, np.max(lhist)*0.3, f"{uq:.1e}", color="black", bbox=bbox, **plot_kwargs)
        ax.text(median, np.max(lhist)*0.3, f"{median:.1e}", bbox=bbox, **plot_kwargs)
        ax.text(np.min(lbins), np.max(lhist)*0.9, name)
    
    ax.set_ylim(0, np.max(lhist)*1.1)

def combine_legends(axes):
    handles = []
    labels = []

    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    return handles, labels

def _format_x(origin, ax):
    format_func = Date2Delta(origin=origin)
    formatter = ticker.FuncFormatter(format_func)
    ax.xaxis.set_major_formatter(formatter)

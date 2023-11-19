import os
import click
import numpy as np
from matplotlib import pyplot as plt

# import own packages
from pymob.utils.store_file import (
    import_package, opt, prepare_casestudy, prepare_scenario, 
    store_sbi_simulations)
from pymob.utils import help

# Prepare priors and simulator =================================================
# training the network. Here I learn the relationship between parameter inout
# and simulation outputs

@click.command()
@click.option("-c", "--case_study", type=str, default="core_daphnia", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="expo_control", 
              help=help.scenario)
def main(case_study, scenario):
    config = prepare_casestudy((case_study, scenario), "sbi.cfg")

    # parse config file
    dataset_kwargs = {k:config["dataset"].getlist(k) for k in config["dataset"].keys()}
    output = config["case-study"].get("output")
    plot_output = config["case-study"].get("plots")

    mod = import_package(package_path=config["case-study"]["package"])
    summary_statistics = getattr(mod.stats, config["sbi"]["summary_stats"])    
    dataset = getattr(mod.data, config["sbi"]["dataset"])
    prior = getattr(mod.prior, config["sbi"]["prior"])
    # Load Estimator and samples (optional) ========================================
    # Plug in Real data ============================================================
    x_experiment = dataset(**dataset_kwargs)
    x_experiment = summary_statistics(x_experiment)
    # x_experiment = x_experiment[~x_experiment.isnan().any(1), :] # exclude rows with nan values

    stats_names = x_experiment.coords["variable"].values

    x_sim = np.loadtxt(os.path.join(output, "x.txt"))
    theta_sim = np.loadtxt(os.path.join(output, "theta.txt"))

    def plot_good_param_space(n_theta, n_summary_stat, xscale="linear"):
        t_s = theta_sim[:, n_theta]
        x_s = x_sim[:, n_summary_stat]
        x_e = x_experiment[:, n_summary_stat].values
        idx_k = np.logical_and(x_s > x_e.min(), x_s < x_e.max())
        thetas_inside_obs_bounds = np.where(idx_k)[0]
        if len(thetas_inside_obs_bounds) == 0:
            xmax = t_s.max()
            xmin = t_s.min()
        else:
            xmax = np.quantile(t_s[thetas_inside_obs_bounds], .95)
            xmin = np.quantile(t_s[thetas_inside_obs_bounds], .05)

        fig, ax = plt.subplots(1,1)
        ax.scatter(t_s, x_s, alpha=.1)
        ax.set_ylim(0, x_e.max()*4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylabel(stats_names[n_summary_stat])
        ax.set_xlabel(prior.keys[n_theta].split(".")[-1])
        ax.set_xscale(xscale)
        ax.hlines([x_e.min(), x_e.max()], [xmin], [xmax], color="tab:orange")
        f = f"{xscale}x__{prior.keys[n_theta].split('.')[-1]}__{stats_names[n_summary_stat]}"
        d = os.path.join(plot_output, "prior_predictive_checks")
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f))
        
    N, K = x_sim.shape
    idx = np.repeat(True, N)
    fig, axes = plt.subplots(K, 1, figsize=(8,8))
    
    for k in range(K):
        x_e = x_experiment[:, k].values
        x_s = x_sim[:, k]

        idx_k = np.logical_and(x_s > x_e.min(), x_s < x_e.max())
        idx = np.logical_and(idx, idx_k)
        
        axes[k].hist(x_s, color="tab:blue")
        ax = axes[k].twinx()
        ax.hist(x_e, color="tab:orange")
        ax.plot(
            np.nan, 
            label=f"{stats_names[k]}, overlap {idx_k.sum()}/{N}", 
            linestyle=""
        )
        ax.legend()

    # calculate the amount of simulation results which are inside the 
    # experimental observations
    n_inside_bounds = idx.sum()

    fig.suptitle(
        f"Number of simulated results inside observed bounds: {n_inside_bounds} " +
        "(complete match)\n"+
        f"see {mod}.stats for the interpretation of the summary statistics"
    )

    if n_inside_bounds < 0.1 * N:
        axes[0].set_title(
            f"WARNING! Simulations inside bounds should be > 10% of sims ({N})",
            color="red"
        )

    fig.savefig(os.path.join(plot_output, "prior_predictive_checks.png"), dpi=300)
    
    Kt = theta_sim.shape[1]
    fig2, axes = plt.subplots(nrows=K, ncols=Kt, figsize=(8,12), sharex="col", sharey="row")
    fig2.text(0.5, 0.04, 'parameter', ha='center')
    fig2.text(0.04, 0.5, 'summary statistic', va='center', rotation='vertical')
    fig2.suptitle(
        "prior ~ simulated summary statistic\n"+
        "the windows should be more or less evenly filled."
    )
    
    for i in range(Kt):
        for j in range(K):
            plot_good_param_space(i, j, xscale="linear")
            plot_good_param_space(i, j, xscale="log")
            y = x_sim[:, j]
            x = theta_sim[:, i]
            axes[j,i].scatter(x, y, alpha=.1)
            axes[j,i].set_xlim(*np.quantile(x[~np.isnan(x)], (0.5,.95)))
            axes[j,i].set_ylim(*np.quantile(y[~np.isnan(y)], (0.5,.95)))
            axes[j,0].set_ylabel(stats_names[j])

        axes[K-1, i].set_xlabel(prior.keys[i].split(".")[-1])

    fig2.savefig(os.path.join(
        plot_output, 
        "prior_predictive_checks_prior_vs_result.png"), dpi=300)



if __name__ == "__main__":
    main()

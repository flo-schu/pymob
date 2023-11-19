import os
import glob
import click
import numpy as np
import xarray as xr

# import own packages
from pymob.utils.store_file import (
    prepare_casestudy, import_package, opt, prepare_scenario)
from pymob.sim.simulation import update_parameters
from pymob.utils import help

@click.command()
@click.option("-c", "--case_study", type=str, default="core_daphnia", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="expo_control", 
              help=help.scenario)
@click.option("-n", "--number_simulations", type=int, default=None)
@click.option("-p", "--posterior_cluster", type=str, default=None)
def main(case_study, scenario, number_simulations, posterior_cluster):
    # parse sbi config file
    config = prepare_casestudy((case_study, scenario), "sbi.cfg")
    sim_conf = prepare_scenario(case_study, scenario)

    # load case study section settings (some parts of the section are populated 
    # automatically by the call to store.prepare_casestudy(...))
    output = config["case-study"].get("output")
    package = config["case-study"].get("package")
    print(f"forwarding output to {output}", flush=True)

    # import case study and load necessary objects
    mod = import_package(package_path=package)
    summary_stats = getattr(mod.stats, "raw")
    generator = getattr(mod.sim, config["sbi"].get("generator", fallback="generator"))

    # other options (override order is always a=command line, b=config file, c=default)
    n_pp = int(opt(a=number_simulations, b=config["sbi"].get("n_posterior_predictions"), c=100))
    pc = opt(a=posterior_cluster, b=config["sbi"].get("posterior_cluster"), c="*")
    obs_interval = config["sbi"].get("posterior_obs_interval", fallback="table")
    observations = config["sbi"].get("posterior_obs")

    # load specific or all clusters of posteriors (if chains have not converged)
    clusters = glob.glob(os.path.join(output, f"posterior_{pc}.nc"))
    print(clusters)
    # adapt simulation config file
    sim_conf["display_plots"] = False
    sim_conf["experiment"]["observation_interval"] = obs_interval
    sim_conf["experiment"]["observations"] = observations.split(" ")
    sim_conf["simulation"]["fail_silently"] = True

    # create necessary directories
    os.makedirs(os.path.join(output, "posterior_predictions"), exist_ok=True)

    for i, filename in enumerate(clusters):
        posterior = xr.load_dataarray(filename)
        
        # set up and run simulation
        print(len(posterior.sample))
        random_samples = np.random.randint(0, len(posterior.sample), n_pp)
        # out = generator(summary=summary_stats, config=sim_conf, init=True)
        for j in random_samples:
            sim_conf = update_parameters(
                sim_conf, posterior.isel(sample=j).values, 
                posterior.parameter.values)
            out = generator(
                summary=summary_stats, config=sim_conf, 
                init=False, return_xarray=True)

            out.to_netcdf(os.path.join(
                output, "posterior_predictions", f"pred_cluster_{i+1}_sample_{j}.nc"
            ))


if __name__ == "__main__":
    main()

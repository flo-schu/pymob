import pickle
import os
import click
import numpy as np

# import own packages
from pymob.utils.store_file import (
    prepare_casestudy, import_package, parse_config_section)
from pymob.utils import help


@click.command()
@click.option("-c", "--case_study", type=str, default="core_daphnia", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="expo_control", 
              help=help.scenario)
@click.option("-w", "--worker", type=str, default="1")
def main(case_study, scenario, worker):
    # parse sbi config file
    config = prepare_casestudy((case_study, scenario), "sbi.cfg")

    output = config["case-study"].get("output")
    mod = import_package(package_path=config["case-study"]["package"])
    dataset = getattr(mod.data, config["sbi"]["dataset"])
    summary_statistics = getattr(mod.stats, config["sbi"]["summary_stats"])    
    mcmc_kwargs = parse_config_section(config["mcmc"], "strint")
    dataset_kwargs = parse_config_section(config["dataset"], "list")

    # Load Estimator and samples (optional) ========================================
    with open(os.path.join(output, "snle", "SNLE_neural_likelihood_estimator.pkl"), "rb") as handle:
        posterior = pickle.load(handle)

    # Plug in Real data ============================================================
    x_experiment = summary_statistics(dataset(**dataset_kwargs))
    num_samples = int(mcmc_kwargs.pop("num_samples", 1000))

    # draw MCMC samples from the likelihood estimator conditioned on the samples
    samples = posterior.sample(
        (num_samples,), 
        x=x_experiment.values, 
        num_chains=1,
        **mcmc_kwargs
    )

    os.makedirs(os.path.join(output, "snle", "posterior_samples"), exist_ok=True)
    np.savetxt(
        os.path.join(output, "snle", "posterior_samples", f"SNLE_mcmc_samples_chain{worker}.txt"),
        samples.detach().numpy()
    )



if __name__ == "__main__":
    main()

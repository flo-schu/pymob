import os
import pickle

import click
import numpy as np
from torch import tensor
from sbi.inference import SNPE, SNLE

from pymob.utils.store_file import import_package, prepare_casestudy
from pymob.utils import help


def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


@click.command()
@click.option("-c", "--case_study", type=str, default="core_daphnia", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="expo_control", 
              help=help.scenario)
@click.option("-i", "--inferer_engine", type=str, default="SNPE")
@click.option("-l", "--split", type=int, default=None)
@click.option("-b", "--training_batch_size", type=int, default=None)
def main(case_study, scenario, inferer_engine, split, training_batch_size):
    # parse sbi config file
    config = prepare_casestudy((case_study, scenario), "sbi.cfg")

    output = config["case-study"].get("output")
    mod = import_package(package_path=config["case-study"]["package"])
    prior = getattr(mod.prior, config["sbi"]["prior"])
 

    # concatenate files or read existing simulation file
    theta_path = os.path.join(output, "theta.txt")
    theta = tensor(np.loadtxt(theta_path, dtype=np.float32))

    x_path = os.path.join(output, "x.txt")
    x = tensor(np.loadtxt(x_path, dtype=np.float32))
    # Train estimator ==============================================================
    # Create inference object: choose method and estimator

    # nsf estimator implements normalizing spline functions (nsf), which 
    # grant the estimator more freedom to fit any distribution
    if split is None:
        split = theta.shape[0]

    if training_batch_size is None:
        training_batch_size = max(1, round_to_multiple(split / 100, 10))

    # initialize density estimator
    if inferer_engine == "SNPE":
        inferer = SNPE(prior, density_estimator="nsf", device="cpu")
        fname = "SNPE_neural_posterior_estimator"  
        inferer = inferer.append_simulations(theta[0:split], x[0:split]) 
        density_estimator = inferer.train(
            training_batch_size=training_batch_size
        )  # Lots of training settings.

        # Build posterior using trained density estimator
        posterior = inferer.build_posterior(density_estimator)

    elif inferer_engine == "SNLE":
        inferer = SNLE(prior, density_estimator="nsf", device="cpu")  
        fname = "SNLE_neural_likelihood_estimator"  
        inferer = inferer.append_simulations(theta[0:split], x[0:split]) 
        density_estimator = inferer.train(
            training_batch_size=training_batch_size
        )  # Lots of training settings.

        # # Build posterior using trained density estimator
        # potential_fn, theta_transform_ = likelihood_estimator_based_potential(
        #     likelihood_estimator=density_estimator, prior=prior, x_o=None
        # )

        # theta_transform = IndependentTransform(prior.transforms[0], 1).inv

        # posterior = MCMCPosterior(
        #     potential_fn=potential_fn,
        #     theta_transform=theta_transform_,
        #     proposal=prior,
        #     method="nuts",
        #     device="cpu",
        #     x_shape=inferer._x_shape
        # )

        posterior = inferer.build_posterior(density_estimator)


    else:
        raise NotImplementedError(f"{inferer_engine} is not implemented choose SNPE or SNLE")

    # with 1 mio datapoints, the network takes approx. 1-2 minutes for one epoch
    # if the inferer takes approx 100-200 epochs to converge it will take max 400 
    # minutes ~ 6-7 hours.
    # CAUTION: Has to be adapted when training set size changes.

    # create dir if not exists
    dirname = os.path.join(output, inferer_engine.lower())
    os.makedirs(dirname, exist_ok=True)

    with open(os.path.join(dirname, f"{fname}.pkl"), "wb") as handle:
        pickle.dump(posterior, handle)

    with open(os.path.join(dirname, f"{inferer_engine}_inferer_logs.pkl"), "wb") as handle:
        pickle.dump(inferer.summary, handle)

if __name__ == "__main__":
    main()

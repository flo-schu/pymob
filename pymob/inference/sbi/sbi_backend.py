import os
import click
import numpy as np
from functools import partial

from pymob.utils.store_file import (
    import_package, opt, prepare_casestudy, prepare_scenario, 
    store_sbi_simulations)
from pymob.utils import help
from pymob.helpers.errors import errormsg
from pymob.sims.simulation import update_parameters
from pymob.objects.housekeeping import LOGGER
from pymob.sims.priors import TransformedMV

# Prepare priors and simulator =================================================
# training the network. Here I learn the relationship between parameter inout
# and simulation outputs

@click.command()
@click.option("-c", "--case_study", type=str, default="core_daphnia", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="expo_control", 
              help=help.scenario)
@click.option("-n", "--number_simulations", type=int, default=1)
@click.option("-w", "--worker", type=str, default="1")
@click.option("-r", "--return_only_simulator", type=bool, default=False)
def main(case_study, scenario, number_simulations, worker, return_only_simulator):
    # parse sbi config file
    config = prepare_casestudy((case_study, scenario), "sbi.cfg")
    sim_conf = prepare_scenario(case_study, scenario)

    # load case study section settings (some parts of the section are populated 
    # automatically by the call to store.prepare_casestudy(...))
    output = config["case-study"].get("output")

    # import case study and load necessary objects
    mod = import_package(package_path=config["case-study"]["package"])
    prior = getattr(mod.prior, config["sbi"]["prior"])
    summary_stats = getattr(mod.stats, config["sbi"]["summary_stats"])
    generator = getattr(mod.sim, config["sbi"].get("generator", fallback="generator"))

    # modify simulation config entries
    sim_conf["simulation"]["logging"] = "ERROR"
    sim_conf["simulation"]["display_plots"] = False
    sim_conf["simulation"]["fail_silently"] = True
    sim_conf["simulation"]["progressbar"] = False
    sim_conf["experiment"]["observation_interval"] = "table"

    output = os.path.join(output, "sims")
        
    def forward(theta, conf, init=False):
        # update config file
        seed = np.random.randint(0, 1e9)
        conf["simulation"]["seed"] = seed
        conf["simulation"]["output"] = os.path.join(output, worker)

        # update parameters
        if not isinstance(theta, np.ndarray):
            theta = theta.detach().numpy()
            
        if len(theta.shape) > 1:
            raise ValueError("sim does not work with input param > 1D")
        
        if isinstance(prior, TransformedMV):
            conf = update_parameters(conf, theta, prior.keys)
        elif isinstance(prior, (list, tuple)):
            conf = update_parameters(conf, theta, [p.name for p in prior])
        else:
            raise NotImplementedError(f"{type(prior)} prior cannot be processed")
        
        # run the simulation
        out = generator(summary=summary_stats, config=conf, init=init, return_xarray=True)

        return out
        
    
    forward_ = partial(forward, conf=sim_conf)

    if return_only_simulator:
        return forward_, prior

    # load sbi relevant packages
    from sbi.inference import simulate_for_sbi
    from pymob.inference.sbi import sbi_utils
    from torch import tensor

    def sbi_forward(theta):
        """thin wrapper around the simulator that returns a tensor from xarray"""
        out = forward_(theta)
        return tensor(out.values)

    prior = sbi_utils.prepare_sbi_prior(prior)
    simulator = sbi_utils.prepare_simulator_for_sbi(sbi_forward, prior)

    # init sim in case needed 
    _ = forward_(prior.sample((1,))[0], init=True)

    # Generate simulations =========================================================
    cpus = int(os.getenv("SLURM_CPUS_PER_TASK", default=1))
    print(f"generating {number_simulations} simulation on {cpus} cores...", flush=True)
    
    theta, x = simulate_for_sbi(
        simulator, prior, 
        num_simulations=number_simulations, 
        num_workers=cpus,
        simulation_batch_size=1, 
        show_progress_bar=True)

    store_sbi_simulations(output, theta, x, simname=worker)
    print(errormsg(
        f"""
        Finished {number_simulations} simulations. {LOGGER.errors} errors occurred. Check
        output directory for simulations that contained errors.
        """
    ))

if __name__ == "__main__":
    main()

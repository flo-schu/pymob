import click

from pymob.utils.store_file import (
    import_package, opt, prepare_casestudy, prepare_scenario)
from pymob.utils import help

@click.command()
@click.option("-c", "--case_study", type=str, default="test_case_study", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="test_scenario", 
              help=help.scenario)
@click.option("-p", "--package", type=str, default="case_studies", 
              help=help.package)
@click.option("-l", "--logging", type=str, default=None)
@click.option("-f", "--logfile", type=str, default=None)
def main(case_study, scenario, package, logging, logfile):
    # TODO: add error messages, in case
    #       - scenario, 
    #       - case_study, 
    #       - package,
    #       - settings.cfg cannot be found
    config = prepare_casestudy(
        case_study=(case_study, scenario), 
        config_file="settings.cfg", 
        pkg_dir=package
    )

    # update parameters from config file if they are specified
    if logging is not None: config.set("case-study", "logging", logging)
    if logfile is not None: config.set("case-study", "logfile", logfile)

    # import package        
    pkg = import_package(package_path=config["case-study"]["package"])
    Simulation = pkg.sim.Simulation

    # create and run simulation
    sim = Simulation(
        config=config
    )
    sim.compute()

    # store and process output
    sim.dump()
    sim.plot()

if __name__ == "__main__":
    main()
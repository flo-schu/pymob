import os
import glob
import click
import xarray as xr

# import own packages
from pymob.utils.store_file import (
    prepare_casestudy, import_package, parse_config_section)
from pymob.utils import help

@click.command()
@click.option("-c", "--case_study", type=str, default="core_daphnia", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="expo_control_2", 
              help=help.scenario)
@click.option("-p", "--posterior_cluster", type=str, default="*")
def main(case_study, scenario, posterior_cluster):

    # parse function parameters
    config = prepare_casestudy((case_study, scenario), "sbi.cfg")

    # import case study and load necessary objects
    mod = import_package(package_path=config["case-study"]["package"])
    plot = getattr(mod.plot, config["sbi"]["posterior_plot"])
    output = config["case-study"]["output"]
    dataset = getattr(mod.data, config["sbi"]["dataset"])
    
    dataset_kwargs = parse_config_section(config["dataset"], "list")
    true_data = dataset(**dataset_kwargs)

    pc = "*" if posterior_cluster == 0 else str(posterior_cluster)

    clusters = glob.glob(os.path.join(output, f"posterior_{pc}.nc"))

    for i in range(1, len(clusters) + 1):
        obsfiles = glob.glob(os.path.join(
            output, "posterior_predictions",f"pred_cluster_{i}_sample_*.nc"
        ))
        obs_list = []        
        for of in obsfiles:
            obs = xr.load_dataarray(of)
            obs_list.append(obs)

        
        # plot the observations
        fig = plot(obs_list, true_data)

        if not isinstance(fig, list):
            fig = [fig]

        for fi, f in enumerate(fig):
            f.savefig(
                os.path.join(config["case-study"]["plots"], 
                f"life_history_cluster_{i}_{fi}.jpg"), dpi=300)

if __name__ == "__main__":
    main()

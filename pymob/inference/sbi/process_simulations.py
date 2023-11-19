import os
import glob
import click
import numpy as np
from collections import Counter

from pymob.utils.store_file import prepare_casestudy
from pymob.utils import help

@click.command()
@click.option("-c", "--case_study", type=str, default="core_daphnia", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="expo_control", 
              help=help.scenario)
def main(case_study, scenario):
    # parse sbi config file
    config = prepare_casestudy((case_study, scenario), "sbi.cfg")
    output = config["case-study"].get("output")

    input_dir = os.path.join(output, "sims")
    
    theta_path = os.path.join(output, "theta.txt")
    theta_files = glob.glob(os.path.join(input_dir, "*/theta.txt"))
    theta = np.concatenate(
        [np.loadtxt(f, dtype=np.float32) for f in theta_files])
    np.savetxt(theta_path, theta)

    x_path = os.path.join(output, "x.txt")
    x_files = glob.glob(os.path.join(input_dir, "*/x.txt"))
    x = np.concatenate(
        [np.loadtxt(f, dtype=np.float32) for f in x_files])
    np.savetxt(x_path, x)

    errors = glob.glob(os.path.join(input_dir, "*/errors/*/"))
    errormessages = []
    infos = []
    for i, err in enumerate(errors):
        try:
            with open(os.path.join(err, "log.txt"), "r") as f:
                last_line = f.readlines()[-1]
            
            if "Mass was not conserved." in last_line:
                msg = "mass conservation error"
                info = last_line.split(" ")[3]
            else:
                msg = last_line
                info = ""

            errormessages.append(msg)
            infos.append(info)
            # conf = read_config(os.path.join(err, "config.json"))
        except FileNotFoundError:
            pass

    error_report = Counter(errormessages)
    with open(os.path.join(output, "error_report.txt"), "w") as report:
        for error, number in error_report.items():
            report.write(f"{error}: {number}\n")

if __name__ == "__main__":
    main()

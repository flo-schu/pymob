import json
from configparser import ConfigParser

CONSTANTS = ConfigParser()
CONSTANTS.read("config/constants.cfg")

def read_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)

def simulation_io_adapter(input_config_file, input_events_file, output_path):
    """
    A simple adapter to adapt the configuration file to be usable with the
    case-study design for the example of experiment based simulation
    """
    
    cfg = read_config(config_file=input_config_file)

    cfg["experiment"]["eventfile"] = input_events_file
    cfg["simulation"]["output"] = output_path

    return cfg
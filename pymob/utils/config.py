import json
import re
import sympy
import warnings
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


def catch_patterns(expression_str):
    # AVOID USING THE DESING OF EXPRESSIONS. JUST USE SYMPY SYNTAX._FlagsType
    # THIS WILL BE MORE STABLE AND EASY TO ADAPT NEW CONCEPTS.
    # THE KEY IS NOT THE EXPRESSION BUT THE LOOKUP
    
    # tries to match array notation [0 1 2]
    pattern = r"\[(\d+(\.\d+)?(\s+\d+(\.\d+)?)*|\s*)\]"
    if re.fullmatch(pattern, expression_str) is not None:
        expression_str = expression_str.replace(" ", ",")
        return f"Array({expression_str})"

    # tries to match array notation [0,1,2]
    pattern = r'\[(\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*|\s*)\]'
    if re.fullmatch(pattern, expression_str) is not None:
        return f"Array({expression_str})"

    return expression_str

def lambdify_expression(expression_str):
    # check for parentheses in expression
    
    expression_str = catch_patterns(expression_str)

    # Parse the expression without knowing the symbol names in advance
    parsed_expression = sympy.sympify(expression_str, evaluate=False)
    free_symbols = tuple(parsed_expression.free_symbols)

    # Transform expression to jax expression
    args = [str(s) for s in free_symbols]
    func = sympy.lambdify(
        args, parsed_expression
    )

    return func, args

def lookup(val, *indexable_objects):
    for obj in indexable_objects:
        if val in obj:
            return obj[val]
        else:
            continue

    return val


def lookup_args(args, *objects_to_search):
    return {k: lookup(k, *objects_to_search) for k in args}
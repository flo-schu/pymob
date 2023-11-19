import numpy as np
import argparse
from pymob.sims.optimization import IndyOptimizer
from pymob.sims.simulation import create_sim, update_parameters
from pymob.utils.store_file import read_config

parser = argparse.ArgumentParser(
    prog="Parameter optimization on observation timeseries of individuals")
parser.add_argument("-s", "--sim_conf", type=str,
                    default="config/parameters/optimization_indy.json", 
                    help="parameter file for the simulation")
parser.add_argument("-o", "--opt_conf", type=str,
                    default="config/parameters/opt_conf_indy_debug.json",
                    help="(hyper)parameter file for the optimization process")
parser.add_argument("-i", "--iterations", type=int, default=0, 
                    help="set iterations of basinhopping optimizer - defaults \
                          to iterations of opt_conf if not set.")
parser.add_argument("-t", "--temperature", type=float, default=0,
                    help="The “temperature” parameter for the accept or reject \
                          criterion. Higher “temperatures” mean that larger jumps\
                          in function value will be accepted. For best results T\
                          should be comparable to the separation (in function value)\
                          between local minima.")
parser.add_argument("-p", "--step", type=float, default=0,
                    help="maximum stepsize used in random displacement")

args = parser.parse_args()


def loss_function(s, use_sim_cols, exp_data, scaler):
    sim_data = s.events.get_tensor()[:, 0, use_sim_cols]

    # calculate the difference between simulation results and
    # experiment results and remove nan values (these result only because
    # of zero values)
    diff = scaler.transform(np.nan_to_num(sim_data, nan=0)) - exp_data

    ssq=np.sum(np.nan_to_num(diff, nan=0)**2)
    return ssq

def iteration(x, config, scales, use_sim_cols, exp_data, scaler, parnames):
    config = update_parameters(config, x, parnames, scales)
    s = create_sim(config)
    s.run(progress_bar=False)
    loss = loss_function(s, use_sim_cols, exp_data, scaler)
    # print("memory:", asizeof.asizeof(s))
    print("loss:", loss)
    return loss

sim_conf = args.sim_conf
opt_conf = read_config(args.opt_conf)

if args.iterations > 0:
    opt_conf["iterations"] = args.iterations

if args.temperature > 0:
    opt_conf["temperature"] = args.temperature

if args.step > 0:
    opt_conf["step"] = args.step

print("using optimization configuration:\n", opt_conf)

io = IndyOptimizer(
    f=iteration, 
    params=opt_conf["parameters"], 
    temperature=opt_conf["temperature"],
    stepsize=opt_conf["step"],
    niter=opt_conf["iterations"],
    interval=opt_conf["interval"],
    sim_config=sim_conf,
    seed=opt_conf["seed"],
    minimizer_kwargs=opt_conf["minimizer_kwargs"],
    optimize_on_sample=opt_conf["sample"],
    optimize_on_features=opt_conf["optimize_on"])

io.optim()
io.plot_results(update_parameters(io.sim_config, io.result.x,
                                  io.parnames, io.parameter_scales))
io.store_data()
# there are still some problems some debugging loops are necessary
# [x] standardize data and results
# [x] make death deterministic
# [x] store results
# [ ] write wrapper for optimization script for calibration on cluster

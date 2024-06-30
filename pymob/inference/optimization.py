raise NotImplementedError(
    "The optimization backend is currently not implemented in pymob"
)

from scipy.optimize import basinhopping
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler
from data.datasets import indy
from pymob.utils.store_file import read_config
from sims.simulation import create_sim, Simulation
from matplotlib import pyplot as plt
import os
import json
import pandas as pd
import numpy as np
import sys
import shutil        


class IndyOptimizer():
    def __init__(
        self, 
        f,
        params,
        niter,
        temperature,
        stepsize,
        interval,
        minimizer_kwargs={},
        sim_config="config/parameters/optimization_indy.json",
        optimize_on_features="all",
        optimize_on_sample=0,
        seed=None
        ):
        self.f = f
        self.parnames = []
        self.x0 = []
        self.bounds = []
        self.niter = niter
        self.temperature = temperature
        self.stepsize = stepsize
        self.interval = interval
        self.steps = []
        self.seed = seed
        self.minimizer_kwargs = minimizer_kwargs
        self.optimize_on_features=optimize_on_features
        self.optimize_on_sample=optimize_on_sample
        self.parse_params(params)
        
        self.data = indy()
        self.hist = {"x":[], "f": [], "a":[]}
        self.result = None
        self.x = None
        self.sim_config = read_config(sim_config)
        self.test_sim = create_sim(self.sim_config)
        self.test_sim.run()

        self.use_sim_cols = self.find_sim_cols()
        self.use_exp_cols = self.find_exp_cols()

        self.scaler = self.construct_scaler()
        self.parameter_scales = self.harmonize_scales()
        self.prepare_data()
        self.copy_config()
        print("--- initialized optimization ---")
        print("optimizing for:", optimize_on_features)
        print("optimizing parameters:", self.parnames)


    class Trap:
        """
        Escape Minimization if the same local minimum is encountered twice in a 
        row (happens due to non-continuous energy landscapes)
        """

        def __init__(self):
            pass
            # Avoid ZeroDivisionError since "MBH can be regarded as a special case
            # of the BH framework with the Metropolis criterion, where temperature
            # T = 0." (Reject all steps that increase energy.)

        def accept_reject(self, energy_new, energy_old):
            """
            if new energy is equal to old energy, reject the test, because the
            minimum was reached before and most of the time means that 
            the optimizer is trapped in a far off region in a low energy landscape
            accept (return true) if energy new is unequal to energy old
            """
            return energy_new != energy_old

        def __call__(self, **kwargs):
            """
            f_new and f_old are mandatory in kwargs
            """
            return bool(self.accept_reject(kwargs["f_new"],
                        kwargs["f_old"]))


    class TakeCustomSteps:
        def __init__(self, stepsize=1, steps=[]):
            self.stepsize = stepsize
            self.rng = np.random.default_rng()
            self.steps = steps

        def prepare_steps(self, x):
            if len(self.steps) == 0:
                return np.array(np.repeat(self.stepsize, len(x)))

            return self.stepsize * np.array(self.steps)

        def __call__(self, x):
            s = self.prepare_steps(x)
            x += self.rng.uniform(-s, +s)
            return x

    def parse_params(self, params):
        # parse initial guess of x
        for key, value in params.items():
            self.parnames.append(key)
            self.x0.append(value[0])

        self.x0 = np.array(self.x0)
        scales = self.harmonize_scales()

        # parse bounds and steps
        for (key, value), scal in zip(params.items(), scales):
            lb = value[1][0] / scal
            ub = value[1][1] / scal
            try:
                s = value[2] / scal
                self.steps.append(s)
            except IndexError:
                # if no steps are used take step to the initial value times stepsize
                s = value[0] / scal * 0.5
                self.steps.append(s)

            self.bounds.append((lb, ub))
        
    def find_exp_cols(self):
        labels = self.data["labels"]["values"]
        if self.optimize_on_features == "all":
            self.optimize_on_features = labels
            return [True] * len(labels)
        return [True if l in self.optimize_on_features else False for l in labels]

    def find_sim_cols(self):
        labels = self.test_sim.events.observations["labels"][1:]
        if self.optimize_on_features == "all":
            self.optimize_on_features = labels
            return [True] * len(labels)
        return [True if l in self.optimize_on_features else False for l in labels]

    def prepare_data(self):
        self.exp_sample = self.data["data"][:, self.optimize_on_sample, self.use_exp_cols]
        # mask = np.all(np.isnan(exp_sample), axis=1)
        # self.exp_sample = exp_sample[~mask]
        self.exp_sample_transformed = self.scaler.transform(self.exp_sample)
        
    def construct_scaler(self):
        # TODO: scaling is still a bit problematic, because they affect the fitting procedure
        #       and different transformers result in different weights
        #       sometimes large offspring data has almost no higher effect over small offspring (yeo-johnson)
        #       data under quantile transformer it is a bit disproportionally high
        #       In addition: scaling under quantile transformer optimizes heavily for 
        #       offspring frequency and not for quantitiy (has almost no effect)
        
        # scaler = QuantileTransformer(random_state=0, output_distribution="normal")
        # scaler = PowerTransformer(method="yeo-johnson")
        # new_shape = (self.data["shape"]["day"] * self.data["shape"]["sample"], sum(self.use_exp_cols))
        # data = self.data["data"][:, :, self.use_exp_cols].reshape(new_shape)
        new_shape = (self.data["shape"]["day"] , sum(self.use_exp_cols))
        data = self.data["data"][:, self.optimize_on_sample, self.use_exp_cols].reshape(new_shape)
        data = np.row_stack((data, np.zeros(sum(self.use_exp_cols))))

        scaler = MinMaxScaler().fit(data)
        return scaler

    def harmonize_scales(self):
        return 10 ** np.ceil(np.log10(self.x0))

    def callback(self, x, f, accepted):
        self.hist["x"].append(x * self.parameter_scales)
        self.hist["f"].append(f)
        self.hist["a"].append(int(accepted))
        # update trajectory as it goes on (helpful if bh optimizer is terminated prematurely)
        self.save_trajectory() 
        print("at minimum %.4f accepted %d" % (f, int(accepted)), x * self.parameter_scales, file=sys.stdout)

    def optim(self):
        config = self.sim_config
        scales = self.parameter_scales
        use_sim_cols = self.use_sim_cols
        scaler = self.scaler
        exp_data = self.exp_sample_transformed
        parnames = self.parnames
        minimizer_kwargs = self.minimizer_kwargs.copy()

        minimizer_kwargs.update({
            "bounds": self.bounds,
            "args": (config, scales, use_sim_cols, exp_data, scaler, parnames)   
        })

        f = self.f
        x0 = self.x0 / scales
        temperature = self.temperature
        stepsize = self.stepsize
        interval = self.interval
        accept_test = self.Trap()
        take_step = self.TakeCustomSteps(stepsize, self.steps)
        niter = self.niter
        callback = self.callback
        seed = self.seed

        self.result = basinhopping(
            f, x0, niter, 
            T=temperature,
            # stepsize=stepsize, # only needed if no custom step taking algorithm was defined
            take_step=take_step,
            interval=interval,
            minimizer_kwargs=minimizer_kwargs,
            callback=callback,  
            accept_test=accept_test,
            disp=True,
            seed=seed,
        )

        self.x = self.result.x * self.parameter_scales

    def copy_config(self):
        path = self.test_sim.directory
        self.test_sim.create_directory()

        # copy config
        sim_conf = self.sim_config
        with open(os.path.join(path, "sim_conf.json"), "w") as f:
            json.dump(sim_conf, f, indent=4)

        # copy eventfile
        eventfile = sim_conf["events"]["eventfile"]
        shutil.copy(eventfile, os.path.join(path, os.path.basename(eventfile)))

    def store_data(self):
        instance = self.__dict__
        data = {}
        keys = [
            "x0", "bounds", "niter", "seed", "temperature", "stepsize",
            "minimizer_kwargs",
            "optimize_on_features", "optimize_on_sample", 
            "sim_config", "parameter_scales", "x"
        ]

        for k in keys:
            values = instance[k]
            if isinstance(values, np.ndarray):
                values = list(values)
            
            data.update({k: values})

        self.test_sim.create_directory()
        with open(os.path.join(self.test_sim.directory, "optim_params.json"), "w") as f:
            json.dump(data, f, indent=4)

        self.save_trajectory()

    def save_trajectory(self):
        self.test_sim.create_directory()
        X = pd.DataFrame.from_records(self.hist["x"], columns=self.parnames)
        F = pd.DataFrame(self.hist["f"], columns=["sse"])
        A = pd.DataFrame(self.hist["a"], columns=["accept"])

        D = pd.concat([X,F,A],axis=1)

        D.to_csv(os.path.join(self.test_sim.directory, "optim_trajectory.csv"), index=False)

    def plot_results(self, sim):
        """
        sim:    simulation instance (which has already run or configuration for sim)
        """
        if not isinstance(sim, Simulation):
            s = create_sim(sim)
            s.run()
        sim_data = s.events.get_tensor()[:, 0, self.use_sim_cols]
        exp_data = self.exp_sample
        feature_names = self.optimize_on_features
        scaler = self.scaler
        xlabels = self.data["labels"]["day"]
        sse = self.result.fun

        plot_fit(exp_data, sim_data, feature_names, scaler, xlabels, sse)
        self.test_sim.create_directory()
        plt.savefig(os.path.join(self.test_sim.directory, "optim_result.png"))

def plot_fit(exp_data, sim_data, feature_names, scaler, xlabels, sse):
    exp_sample_transformed = scaler.transform(exp_data)
    sim_sample_transformed = scaler.transform(
        np.nan_to_num(sim_data, nan=0))
    diff = exp_sample_transformed - sim_sample_transformed
    n_features = len(feature_names)
    sse_contrib = np.sum(np.nan_to_num(diff**2, nan=0), axis=0)
    fig, axes = plt.subplots(ncols=n_features, nrows=1, figsize=(n_features*4, 5))

    for i, ax, label in zip(range(sim_data.shape[1]), axes, feature_names):
        ax.scatter(xlabels, np.nan_to_num(
            sim_data[:, i]), label="simulation", alpha=.5)
        ax.scatter(xlabels, exp_data[:, i], label="experiment", alpha=.5)

        if label == "survival":
            ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(label)
        ax.set_xlabel("time [days]")

        ax.vlines(x=xlabels,
                    ymin=np.nan_to_num(sim_data[:, i]),
                    ymax=exp_data[:, i],
                    linestyle="--", color="black")

        # add sse contributions
        ax.text(0.99, 0.99,
            s="sse: {}".format(round(sse_contrib[i], 2)),
                transform=ax.transAxes, va="top", ha="right")

        if i == 0:
            ax.text(0.0, 1.01, s="overall error: {}".format(sse),
                    transform=ax.transAxes)

    axes[0].legend()


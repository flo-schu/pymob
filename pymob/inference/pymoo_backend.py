import json
import numpy as np

import pathos.multiprocessing as mp
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from pymob import SimulationBase


class PymooBackend:
    def __init__(
        self, 
        simulation: SimulationBase,
    ):
        self.simulation = simulation
        self.pool: mp.Pool = None
        self.distance_function = self.distance_function_parser()
        self.transform = self.variable_parser()
        self.problem = OptimizationProblem(backend=self)

    @property
    def population_size(self):
        return self.simulation.config.getint("inference.pymoo", "population_size", fallback=100)
    
    @property
    def seed(self):
        return self.simulation.config.getint("inference.pymoo", "seed", fallback=1)
        
    @property
    def max_nr_populations(self):
        return self.simulation.config.getint("inference.pymoo", "max_nr_populations", fallback=50)
    
    @property
    def xtol(self):
        return self.simulation.config.getfloat("inference.pymoo", "xtol", fallback=0.0005)
    
    @property
    def ftol(self):
        return self.simulation.config.getfloat("inference.pymoo", "ftol", fallback=0.005)
    
    @property
    def cvtol(self):
        return self.simulation.config.getfloat("inference.pymoo", "cvtol", fallback=1e-8)
    
    @property
    def verbose(self):
        return self.simulation.config.get("inference.pymoo", "verbose", fallback=True)
    
    def distance_function_parser(self):
        def f(x):
            Y = self.simulation.evaluate(theta=x)
            obj_name, obj_value = self.simulation.objective_function(results=Y)
            return obj_value
        return f

    def variable_mapper(self, x):
        names = self.simulation.model_parameter_names
        return {n:x_i for n, x_i in zip(names, x)}

    def variable_parser(self):
        variables = self.simulation.free_model_parameters

        bounds = []
        names = []
        for v in variables:
            bounds.append([v.min, v.max])
            names.append(v.name)

        bounds = np.array(bounds).T
        scaler = MinMaxScaler().fit(bounds)

        # check that parameter values and names match, so variable_mapper 
        # can be safely called
        assert names == self.simulation.model_parameter_names

        def transform(X_scaled):
            X = scaler.inverse_transform(X_scaled)
            return map(self.variable_mapper, X)

        return transform
    
    def store_results(self):
        res = self.result

        params = list(self.transform(X_scaled=np.array(res.X, ndmin=2)) )[0]

        res_dict = {
            "f": res.f,
            "cv": res.cv,
            "X": params
        }
        
        with open(f"{self.simulation.output_path}/pymoo_params.json", "w") as fp:
            json.dump(res_dict, fp, indent=4)

    def run(self):
        """Implements the parallelization in pymoo"""
        n_cores = self.simulation.n_cores
        print(f"Using {n_cores} CPU cores")

        if n_cores == 1:
            self.optimize()
        elif n_cores > 1:
            with mp.ProcessingPool(n_cores) as pool:
                self.pool = pool  # assign pool so it can be accessed by _evaluate
                self.optimize()


    def optimize(self):

        reference_directions = get_reference_directions(
            name="energy", 
            n_dim=self.simulation.n_objectives, 
            n_points=self.population_size,
            seed=self.seed + 1
        )

        algorithm = UNSGA3(
            ref_dirs=reference_directions,
            pop_size=self.population_size
        )

        termination = DefaultMultiObjectiveTermination(
            xtol=self.xtol,
            cvtol=self.cvtol,
            ftol=self.ftol,
            n_max_gen=self.max_nr_populations,
        )

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(
            problem=self.problem, 
            termination=termination,
            seed=self.seed, 
            verbose=True,
        )


        # until the algorithm has no terminated
        while algorithm.has_next():

            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()

            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            algorithm.evaluator.eval(self.problem, pop)

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

            # self.post_processing(pop=pop)

        # obtain the result objective from the algorithm
        res = algorithm.result()

        # save the results
        self.result = res
        self.store_results()

    def post_processing(self, pop):
        F = pop.get("F")
        X = pop.get("X")

        f_min = F.min()
        x_min = X[np.where(F[:,0] == f_min)]
        # print(
        #     f"Generation {gen}: f_min={f_min}, x_min={x_min}",
        #     flush=True
        # )

    def load_results(self):
        with open(f"{self.simulation.output_path}/pymoo_params.json", "r") as fp:
            self.result = json.load(fp)
            
    def plot_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={}, 
            upscale_x=True
        ):
        obs = self.simulation.observations.sel(subset)
        x_old = self.simulation.coordinates[x_dim].copy()
        
        nan_obs = obs[data_variable].isnull().all(dim=x_dim)

        if x_dim is not None:    
            if upscale_x:
                self.simulation.coordinates[x_dim] = np.linspace(
                    x_old.min(),
                    x_old.max(),
                    1000
                )
            else:
                raise NotImplementedError("x_new must be a 1D np.ndarray")
            
        Y = self.simulation.evaluate(self.result["X"])
        results = self.simulation.results_to_df(results=Y)

        if ax is None:
            ax = plt.subplot(111)
        
        ax.plot(
            results[x_dim].values, 
            results[data_variable].values.T[:, ~nan_obs], 
            color="black", lw=.8
        )

        ax.plot(
            obs[x_dim].values, 
            obs[data_variable].values[:, ~nan_obs], 
            marker="o", ls="", ms=3
        )
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)

        # restore old coordinates
        self.simulation.coordinates[x_dim] = x_old

        return ax


class OptimizationProblem(Problem):
    def __init__(self, backend: PymooBackend, **kwargs):
        self.backend = backend
        self.simulation = backend.simulation
        
        super().__init__(
            n_var=self.simulation.n_free_parameters, 
            n_obj=self.simulation.n_objectives, 
            n_ieq_constr=0, 
            xl=0.0, 
            xu=1.0,
            elementwise_evaluation=False,
            **kwargs
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X_original_scale = self.backend.transform(X_scaled=X)       

        if self.backend.pool is None:
            F = list(map(self.backend.distance_function, X_original_scale))
        else:
            F = self.backend.pool.map(self.backend.distance_function, X_original_scale)

        out["F"] = np.array(F)
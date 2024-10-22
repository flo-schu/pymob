from typing import Literal, Dict, Optional, List

import xarray as xr
import arviz as az
import numpy as np

from pymob.sim.config import Config
from matplotlib import pyplot as plt

class SimulationPlot:
    def __init__(
        self, 
        observations,
        coordinates,
        config: Config,
        rows: Optional[List[str]] = None,
        columns: Optional[str] = None,
        sharex=True,
        sharey="rows",
    ):
        self.observations: xr.Dataset = observations
        self.coordinates: Dict = coordinates
        self.config: Config = config

        if rows is None:
            self.rows = self.config.data_structure.data_variables

        if columns is None:
            self.columns = []

        self.sharex = sharex
        self.sharey = sharey

        self.create_figure()


    def create_figure(self):
        r = len(self.rows)
        c = len(self.columns)
        self.fig, self.axes = plt.subplot(
            r, c,
            figsize=(5+(c-1*2), 3+(r-1)*2),
            sharex=self.sharex,
            sharey=self.sharey,
        )


    @staticmethod
    def plot_predictions(
            observations,
            predictions,
            data_variable: str,
            x_dim: str, 
            ax=None, 
            plot_preds_without_obs=False,
            subset={},
            mode: Literal["mean+hdi", "draws"]="mean+hdi",
            plot_options: Dict={"obs": {}, "pred_mean": {}, "pred_draws": {}, "pred_hdi": {}},
            prediction_data_variable: Optional[str] = None,
        ):
        # filter subset coordinates present in data_variable
        subset = {k: v for k, v in subset.items() if k in observations.coords}
        
        if prediction_data_variable is None:
            prediction_data_variable = data_variable

        # select subset
        if prediction_data_variable in predictions:
            preds = predictions.sel(subset)[prediction_data_variable]
        else:
            raise KeyError(
                f"{prediction_data_variable} was not found in the predictions "+
                "consider specifying the data variable for the predictions "+
                "explicitly with the option `prediction_data_variable`."
            )
        try:
            obs = observations.sel(subset)[data_variable]
        except KeyError:
            obs = preds.copy().mean(dim=("chain", "draw"))
            obs.values = np.full_like(obs.values, np.nan)
        
        # stack all dims that are not in the time dimension
        if len(obs.dims) == 1:
            # add a dummy batch dimension
            obs = obs.expand_dims("batch")
            obs = obs.assign_coords(batch=[0])

            preds = preds.expand_dims("batch")
            preds = preds.assign_coords(batch=[0])


        stack_dims = [d for d in obs.dims if d not in [x_dim, "chain", "draw"]]
        obs = obs.stack(i=stack_dims)
        preds = preds.stack(i=stack_dims)
        N = len(obs.coords["i"])
            
        hdi = az.hdi(preds, .95)[f"{prediction_data_variable}"]

        if ax is None:
            ax = plt.subplot(111)
        
        y_mean = preds.mean(dim=("chain", "draw"))

        for i in obs.i:
            if obs.sel(i=i).isnull().all() and not plot_preds_without_obs:
                # skip plotting combinations, where all values are NaN
                continue
            
            if mode == "mean+hdi":
                kwargs_hdi = dict(color="black", alpha=0.1)
                kwargs_hdi.update(plot_options.get("pred_hdi", {}))
                ax.fill_between(
                    preds[x_dim].values, *hdi.sel(i=i).values.T, # type: ignore
                    **kwargs_hdi
                )

                kwargs_mean = dict(color="black", lw=1, alpha=max(1/N, 0.05))
                kwargs_mean.update(plot_options.get("pred_mean", {}))
                ax.plot(
                    preds[x_dim].values, y_mean.sel(i=i).values, 
                    **kwargs_mean
                )
            elif mode == "draws":
                kwargs_draws = dict(color="black", lw=0.5, alpha=max(1/N, 0.05))
                kwargs_draws.update(plot_options.get("pred_draws", {}))
                ys = preds.sel(i=i).stack(sample=("chain", "draw"))
                ax.plot(
                    preds[x_dim].values, ys.values, 
                    **kwargs_draws
                )
            else:
                raise NotImplementedError(
                    f"Mode '{mode}' not implemented. "+
                    "Choose 'mean+hdi' or 'draws'."
                )

            kwargs_obs = dict(marker="o", ls="", ms=3, color="tab:blue")
            kwargs_obs.update(plot_options.get("obs", {}))
            ax.plot(
                obs[x_dim].values, obs.sel(i=i).values, 
                **kwargs_obs
            )
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)

        return ax
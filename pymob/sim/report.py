import os
from functools import wraps
from typing import List, Dict, Optional, Literal, Callable
import inspect
import subprocess
import warnings

import numpy as np
import arviz as az
import pandas as pd
from matplotlib import pyplot as plt
from pymob.sim.config import Config
from pymob.inference.analysis import create_table, log, nrmse, bic, log_lik, plot_trace, plot_pairs

def reporting(method):
    @wraps(method)
    def _inner(self: "Report", *method_args, **method_kwargs):
        report_name = method.__name__
        head = f"## Report: {report_name.replace('_', ' ').capitalize()}"
        # report unless the report is listed in the config file as skipped
        if getattr(self.rc, report_name, True):
            try:
                self._write(head + " ✓")
                out = method(self, *method_args, **method_kwargs)
                self.status.update({report_name: True})
                if out is not None:
                    self._write(
                        "Report '{r}' was successfully generated and saved in '{o}'".format(
                            r=report_name, o=out
                        )
                    )
                return out

            except Exception as e:
                if self.rc.debug_report:
                    raise e
                self._write(head + " ✕")
                self._write("Report '{r}' was not executed successfully".format(
                    r=report_name
                ))
                self.status.update({report_name: False})
        else:
            self._write(head + " ⏭")
            self._write("Report '{r}' was skipped".format(
                r=report_name
            ))
            pass
    return _inner


def _nrmse_from_idata(
    idata: az.InferenceData, 
    data_vars: Optional[List[str]] = None, 
    use_predictions: bool = True,
    obs_transform_funcs: Dict[str, Callable] = {},
    nrmse_mode: Literal["range", "mean", "iqr"] = "range",
):
    if data_vars is None:
        data_vars = list(idata.observed_data.data_vars.keys())

    nrmse_data_vars = {}
    for dv in data_vars:
        if use_predictions:
            x_0 = idata.posterior_predictive[dv]
            x = idata.observed_data[dv]
        else:
            x_0 = idata.posterior_model_fits[dv]
            obs_transform_func = obs_transform_funcs.get(dv, lambda x: x)
            x = obs_transform_func(idata.observed_data[dv])

        nrmse_dv = nrmse(x_0, x, mode=nrmse_mode)
        nrmse_data_vars.update({dv: {
            "NRMSE": nrmse_dv.mean().values, 
        }})

        if "chain" in nrmse_dv.coords and "draw" in nrmse_dv.coords:
            # nrmse_data_vars[dv].update({
            #     "NRMSE (hdi 95%)": az.hdi(nrmse_dv, hdi_prob=.95)[dv].values
            # })
            _nrmse_hdi = az.hdi(nrmse_dv, hdi_prob=.95)[dv].values
            nrmse_data_vars[dv].update({
                "NRMSE (95%-hdi[lower])": _nrmse_hdi[0],
                "NRMSE (95%-hdi[upper])": _nrmse_hdi[1]
            })
    
    nrmse_data_vars["model"] = np.nan   

    return pd.DataFrame(nrmse_data_vars).T

def _loglik_from_idata(
    idata: az.InferenceData, 
    data_vars: Optional[List[str]] = None
):
    if data_vars is None:
        data_vars = list(idata.observed_data.data_vars.keys())

    _loglik_sum = log_lik(idata.log_likelihood.to_array().sum("variable"))

    loglik_data_vars = {}
    for dv in data_vars:
        _loglik_dv = log_lik(idata.log_likelihood[dv])

        loglik_data_vars.update({dv: {
            "Log-Likelihood": _loglik_dv.mean().values, 
        }})


        if "chain" in _loglik_dv.coords and "draw" in _loglik_dv.coords:
            # loglik_data_vars[dv].update({
            #     "Log-Likelihood (hdi 95%)": az.hdi(_loglik_dv, hdi_prob=.95)[dv].values
            # })
            _loglik_hdi = az.hdi(_loglik_dv, hdi_prob=.95)[dv].values
            loglik_data_vars[dv].update({
                "Log-Likelihood (95%-hdi[lower])": _loglik_hdi[0],
                "Log-Likelihood (95%-hdi[upper])": _loglik_hdi[1]
            })


    loglik_data_vars["model"] = {"Log-Likelihood": _loglik_sum.mean().values}

    if "chain" in _loglik_sum.coords and "draw" in _loglik_sum.coords:
        # loglik_data_vars["model"]["Log-Likelihood (hdi 95%)"] = az.hdi(_loglik_dv, hdi_prob=.95)[dv].values
        _loglik_sum_hdi = az.hdi(_loglik_sum, hdi_prob=.95).x.values
        loglik_data_vars["model"].update({
            "Log-Likelihood (95%-hdi[lower])": _loglik_sum_hdi[0],
            "Log-Likelihood (95%-hdi[upper])": _loglik_sum_hdi[1]
        })

    return pd.DataFrame(loglik_data_vars).T

def _bic_from_idata(
    idata: az.InferenceData, 
    free_params: List[str],
    data_vars: Optional[List[str]] = None,
):
    """calculate the BIC for az.InferenceData. The function will average over
    all samples from the markov chain
    """
    
    # this may be computationally more efficient
    # log_likelihood = idata.log_likelihood.mean(("chain", "draw")).sum().to_array().sum()

    L = log_lik(idata.log_likelihood.to_array().sum("variable"))
    k = idata.posterior[free_params].mean(("chain", "draw")).count().to_array().sum()
    N = idata.observed_data.count()
    n = N.to_array().sum().values
    
    _bic = k * np.log(n) - 2 * L

    summary = {dv: {"n (data)": N[dv].values}  for dv in data_vars}
    summary.update({"model": {
        "n (data)": n, 
        "k (parameters)": k.values, 
        "BIC": _bic.mean().values
    }})

    if "chain" in _bic.coords and "draw" in _bic.coords:
        _bic_hdi = az.hdi(_bic, hdi_prob=.95).x.values
        summary["model"].update({
            "BIC (95%-hdi[lower])": _bic_hdi[0],
            "BIC (95%-hdi[upper])": _bic_hdi[1]
        })
        
    return pd.DataFrame(summary).T


class Report:
    """Creates a configurable report. To select which items to report and
    to fine-tune the report settings, modify the settings in `config.report`.
    """
    obs_transform_funcs = {}
    def __init__(self, config: Config, backend: type):
        self.config = config
        self.backend = backend
        self.rc = config.report
        self.file = os.path.join(self.config.case_study.output_path, "report.md")

        self.preamble()

        self.status = {}
        self._label = "--".join([
            "{placeholder}", 
            self.config.case_study.name.replace('_','-'),
            self.config.case_study.scenario.replace('_','-')
        ])

    def __repr__(self):
        return "Report(case_study={c}, scenario={s})".format(
            c=self.config.case_study.name, 
            s=self.config.case_study.scenario,
        )

    def _write(self, msg, mode="a", newlines=1):
        log(msg=msg, out=self.file, newlines=newlines, mode=mode)


    def compile_report(self):
        wd = os.getcwd()
        os.makedirs(os.path.join(self.config.case_study.output_path, "reports"), exist_ok=True)
        os.chdir(self.config.case_study.output_path)
        
        try:
            if self.rc.pandoc_output_format == "html":
                out = self._pandoc_to_html()
            elif self.rc.pandoc_output_format == "latex-si":
                out = self._pandoc_to_latex_si()
            elif self.rc.pandoc_output_format == "latex":
                out = self._pandoc_to_latex_standalone()
            elif self.rc.pandoc_output_format == "pdf":
                out = self._pandoc_to_pdf()
                if out.returncode != 0:
                    warnings.warn("Error compiling report to pdf. Do you have latex installed?")
            else:
                warnings.warn(
                    "There was an error compiling the report from report.md to the desired output format! "+ 
                    f"The `pandoc_output_format`: {self.rc.pandoc_output_format} "+
                    "is not defined. Use one of: html, latex, latex-si, pdf. "+
                    "E.g.: `config.report.pandoc_output_format = html`",
                    category=UserWarning
                )

            if out.returncode != 0:
                warnings.warn(
                    "There was an error compiling the report from report.md to the desired output format! "+ 
                    f"Process exited with return code {out.returncode} using the following arguments: "+
                    f"{' '.join(out.args)}",
                    category=UserWarning
                )
        except FileNotFoundError:
            warnings.warn(
                "There was an error compiling the report! "
                "Pandoc seems not to be installed. Make sure to install pandoc on your "+
                "system. Install with: `conda install -c conda-forge pandoc` "+
                "(https://pandoc.org/installing.html)",
                category=UserWarning
            )

        os.chdir(wd)

    def _pandoc_to_html(self):
        os.chdir("reports")
        return subprocess.run([
            "pandoc",        
            "--resource-path=..",
            f"--extract-media=media/{self.config.case_study.name}_{self.config.case_study.scenario}",
            "--standalone",
            f"--output={self.config.case_study.name}_{self.config.case_study.scenario}.html",
            "../report.md"
        ])

    def _pandoc_to_latex_standalone(self):
        os.chdir("reports")
        return subprocess.run([
            "pandoc",        
            "--resource-path=..",
            f"--extract-media=media/{self.config.case_study.name}_{self.config.case_study.scenario}",
            "--standalone",
            f"--output={self.config.case_study.name}_{self.config.case_study.scenario}.tex",
            "../report.md"
        ])

    def _pandoc_to_latex_si(self):
        return subprocess.run([
            "pandoc",        
            "--resource-path=.",
            f"--extract-media=reports/media/{self.config.case_study.name}_{self.config.case_study.scenario}",
            f"--output=reports/{self.config.case_study.name}_{self.config.case_study.scenario}.tex",
            "report.md"
        ])

    def _pandoc_to_pdf(self):
        return subprocess.run([
            "pandoc",        
            "--resource-path=.",
            f"--output=reports/{self.config.case_study.name}_{self.config.case_study.scenario}.pdf",
            "--pdf-engine=xelatex",
            "report.md"
        ])

    def preamble(self):
        # metadata block
        self._write("---", newlines=0, mode="w")
        self._write("maxwidth: 80%", newlines=0)
        self._write("---")

        title="{header}\n{underline}".format(
            header=str(self),
            underline='=' * len(str(self))
        )
        self._write(title)

        self._write("+ Using `{c}=={v}`".format(
            c=self.config.case_study.name,
            v=self.config.case_study.version,
        ), newlines=0)

        self._write("+ Using `pymob=={v}`".format(
            v=self.config.case_study.pymob_version,
        ), newlines=0)

        self._write("+ Using backend: `{b}`".format(
            b=self.backend.__name__,
        ), newlines=0)

        self._write("+ Using settings: `{s}`".format(
            s=os.path.join(self.config.case_study.scenario_path, "settings.cfg"),
        ), newlines=1)


    @reporting
    def table_parameter_estimates(self, posterior, indices):

        if self.rc.table_parameter_estimates_with_batch_dim_vars:
            var_names = {
                k: k for k, v in self.config.model_parameters.free.items()
            }
        else:
            var_names = {
                k: k for k, v in self.config.model_parameters.free.items()
                if self.config.simulation.batch_dimension not in v.dims
            }

        var_names.update(self.rc.table_parameter_estimates_override_names)

        if len(self.rc.table_parameter_estimates_exclude_vars) > 0:
            self._write(f"Excluding parameters: {self.rc.table_parameter_estimates_exclude_vars} for meaningful visualization")

            var_names = {
                k: k for k, v in var_names.items() 
                if k not in self.rc.table_parameter_estimates_exclude_vars
            }

        tab_report = create_table(
            posterior=posterior,
            vars=var_names,
            error_metric=self.rc.table_parameter_estimates_error_metric,
            significant_figures=self.rc.table_parameter_estimates_significant_figures,
            nesting_dimension=indices.keys(),
            parameters_as_rows=self.rc.table_parameter_estimates_parameters_as_rows,
        )

        # rewrite table in the desired output format
        tab = create_table(
            posterior=posterior,
            vars=var_names,
            error_metric=self.rc.table_parameter_estimates_error_metric,
            significant_figures=self.rc.table_parameter_estimates_significant_figures,
            fmt=self.rc.table_parameter_estimates_format,
            nesting_dimension=indices.keys(),
            parameters_as_rows=self.rc.table_parameter_estimates_parameters_as_rows,
        )

        self._write_table(tab=tab, tab_report=tab_report, label_insert="Parameter estimates")


    def _write_table(
        self, 
        tab: pd.DataFrame, 
        tab_report: Optional[pd.DataFrame] = None, 
        label_insert: Optional[str] = ""
    ):
        if tab_report is None:
            tab_report = tab
            
        self._write(tab_report.reset_index().to_markdown())

        safe_string_insert = label_insert.lower().replace(" ", "_").replace("$", "")
        if self.rc.table_parameter_estimates_format == "latex":
            table_latex = tab.to_latex(
                multicolumn_format="c",
                position="htb",
                float_format=f"%.{self.rc.table_parameter_estimates_significant_figures}g",
                escape=False,
                caption=(
                    f"{label_insert} of the {self.config.case_study.name.replace('_','-')}"+
                    f"({self.config.case_study.scenario.replace('_','-')}) model."
                ),
                label=self._label.format(placeholder=f"tab:{safe_string_insert}")
            )

            out = f"{self.config.case_study.output_path}/report_table_{safe_string_insert}.tex"
            with open(out, "w") as f:
                f.writelines(table_latex)

        elif self.rc.table_parameter_estimates_format == "csv":
            out = f"{self.config.case_study.output_path}/report_table_{safe_string_insert}.csv"
            tab.to_csv(out)


        elif self.rc.table_parameter_estimates_format == "tsv":
            out = f"{self.config.case_study.output_path}/report_table_{safe_string_insert}.tsv"
            tab.to_csv(out, sep="\t")

        return out

    @reporting
    def model(self, model: Callable, post_processing: Optional[Callable]):
        self._write("### Model")
        self._write(f"```python\n{inspect.getsource(model)}\n```")
        
        if post_processing is not None:
            self._write("### Solver post processing")
            self._write(f"```python\n{inspect.getsource(post_processing)}\n```")
        
        if self.backend.__name__ == "NumpyroBackend":
            self._write("### Probability model")
            self._write("![Directed acyclic graph (DAG) of the probability model.](probability_model.png)")

    @reporting
    def parameters(self, model_parameters):
        self._write("### $x_{in}$")
        if "x_in" in model_parameters:
            if self.rc.parameters_format == "xarray":
                self._write(model_parameters["x_in"]._repr_html_())
            elif self.rc.parameters_format == "pandas":
                # TODO: In future: Iterate over variables in x_in, then make a wide table
                #       batch_dimension x x_dimension for each additional index coordinate
                try:
                    self._write(model_parameters["x_in"].to_pandas().to_markdown())
                except ValueError:
                    self._write(
                        "$x_{in}$ is to complex to represent as dataframe. Using xarray "+
                        "representation instead. Compile to html with pandoc to render "+
                        "report"
                    )
                    self._write(model_parameters["x_in"]._repr_html_())

        else:
            self._write("No model input")

        
        self._write("### $y_0$")
        if "y0" in model_parameters:
            if self.rc.parameters_format == "xarray":
                self._write(model_parameters["y0"]._repr_html_())
            elif self.rc.parameters_format == "pandas":
                try:
                    self._write(model_parameters["y0"].to_pandas().to_markdown())
                except ValueError:
                    self._write(
                        "$y_{0}$ is to complex to represent as dataframe. Using xarray "+
                        "representation instead. Compile to html with pandoc to render "+
                        "report"
                    )
                    self._write(model_parameters["y0"]._repr_html_())
        else:
            self._write("No starting values")


        self._write("### Free parameters", newlines=2)
        for key, param in self.config.model_parameters.free.items():
            if param.prior is not None:
                prior = param.prior.model_ser()
                dims = param.dims
                self._write(
                    "+ {key} $\sim$ {prior}".format(
                        key=key, prior=prior
                    ).replace(")", ",dims={dims})".format(dims=dims)), 
                    newlines=0
                )
        
        self._write("\n\n### Fixed parameters", newlines=2)
        for key, param in self.config.model_parameters.fixed.items():
            self._write(
                "+ {key} $=$ {value}, dims={dims}".format(
                    key=key, 
                    value=param.value,
                    dims=param.dims
                ), 
                newlines=0
            )

        self._write("")

    @reporting
    def goodness_of_fit(self, idata):
        free_params = [
            k for k, v in self.config.model_parameters.free.items()
            if self.config.simulation.batch_dimension not in v.dims
        ]

        _nrmse = _nrmse_from_idata(
            idata=idata, 
            data_vars=self.config.data_structure.observed_data_variables, 
            use_predictions=self.rc.goodness_of_fit_use_predictions,
            obs_transform_funcs=self.obs_transform_funcs,
            nrmse_mode=self.rc.goodness_of_fit_nrmse_mode,
        )
        
        _loglik = _loglik_from_idata(
            idata=idata, 
            data_vars=self.config.data_structure.observed_data_variables, 
        )

        _bic = _bic_from_idata(
            idata=idata,
            free_params=free_params,
            data_vars=self.config.data_structure.observed_data_variables, 
        )

        df = pd.concat([_nrmse.T, _loglik.T, _bic.T])
        out = f"{self.config.case_study.output_path}/goodness_of_fit.csv"
        df.to_csv(out)

        self._write(df.to_markdown())
        return out

    @reporting
    def posterior(self, posterior):
        """Much of it included in the parameter estimates and may add to the confusion"""
        self._write("The posterior is saved as an [`xarray`](https://docs.xarray.dev/en/stable/index.html) object that contains draws from the estimated distributions or point estimates in the case of optimizations")
        
        self._write("### Posterior mean")
        self._write(posterior.mean(("chain", "draw"))._html_repr_())


    @reporting
    def diagnostics(self, idata):

        if self.rc.diagnostics_with_batch_dim_vars:
            var_names = {
                k: k for k, v in self.config.model_parameters.free.items()
            }
        else:
            var_names = {
                k: k for k, v in self.config.model_parameters.free.items()
                if self.config.simulation.batch_dimension not in v.dims
            }

        if len(self.rc.diagnostics_exclude_vars) > 0:
            self._write(f"Excluding parameters: {self.rc.diagnostics_exclude_vars} for meaningful visualization")

            var_names = {
                k: k for k, v in var_names.items() 
                if k not in self.rc.diagnostics_exclude_vars
            }


        out = self.config.case_study.output_path

        _, out_pairs = plot_pairs(
            idata=idata, var_names=list(var_names.keys()), output=os.path.join(out, "posterior_pairs.png")
        )
        plt.close()
        self._write("![Paired parameter estimates](posterior_pairs.png)")

        _, out_trace = plot_trace(
            idata=idata, var_names=list(var_names.keys()), output=os.path.join(out, "posterior_trace.png")
        )
        plt.close()
        
        if self.backend.__name__ == "NumpyroBackend":
            if self.config.inference_numpyro.kernel.lower() != "mcmc":
                msg = "Psuedo trace, generated for draws from the optimized SVI distribution"
            else:
                msg = "Traceplot, generated from MCMC draws. "
        self._write(f"![{msg}](posterior_trace.png)")

        return out_pairs, out_trace

    def additional_reports(self, sim):
        pass
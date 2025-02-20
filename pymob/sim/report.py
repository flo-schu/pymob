import os
from functools import wraps

from pymob.sim.config import Config
from pymob.inference.analysis import create_table, log

def reporting(method):
    @wraps(method)
    def _inner(self: "Report", *method_args, **method_kwargs):
        report_name = method.__name__
        head = f"## Report: {report_name.replace('_', ' ').capitalize()}"
        if self.rc[report_name]:
            try:
                self._write(head + " ✅")
                out = method(self, *method_args, **method_kwargs)
                self.status.update({report_name: True})
                self._write(
                    "Report '{r}' was successfully generated and saved in '{o}'".format(
                        r=report_name, o=out
                    )
                )
                return out

            except:
                self._write(head + " ❌")
                self._write("Report '{r}' was not executed successfully".format(
                    r=report_name
                ))
                self.status.update({report_name: False})
        else:
            self._write(head + " ⏩")
            self._write("Report '{r}' was skipped".format(
                r=report_name
            ))
            pass
    return _inner


class Report:
    """Creates a configurable report. To select which items to report and
    to fine-tune the report settings, modify the settings in `config.report`.
    """
    def __init__(self, config: Config):
        self.config = config
        self.rc = config.report
        self.file = os.path.join(self.config.case_study.output_path, "report.md")
        
        self.preamble()

        self.status = {}

    def __repr__(self):
        return "Report(case_study={c}, scenario={s})".format(
            c=self.config.case_study.name, 
            s=self.config.case_study.scenario,
        )

    def _write(self, msg, mode="a", newlines=1):
        log(msg=msg, out=self.file, newlines=newlines, mode=mode)

    def preamble(self):
        msg="{header}\n{underline}".format(
            header=str(self),
            underline='=' * len(str(self))
        )
        self._write(msg, mode="w")

        self._write("Using {c}=={v}".format(
            c=self.config.case_study.name,
            v=self.config.case_study.version,
        ), newlines=0)

        self._write("Using pymob=={v}".format(
            v=self.config.case_study.pymob_version,
        ))

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

        tab = create_table(
            posterior=posterior,
            vars=var_names,
            error_metric=self.rc.table_parameter_estimates_error_metric,
            nesting_dimension=indices.keys(),
            parameters_as_rows=self.rc.table_parameter_estimates_parameters_as_rows,
        )

        if self.rc.table_parameter_estimates_format == "latex":
            table_latex = tab.to_latex(
                float_format="%.2f",
                caption=(
                    f"Parameter estimates of the {self.config.case_study.name}"+
                    f"({self.config.case_study.scenario}) model."
                ),
                label=f"tab:parameters-{self.config.case_study.name}__{self.config.case_study.scenario}"
            )

            out = f"{self.config.case_study.output_path}/report_table_parameter_estimates.tex"
            with open(out, "w") as f:
                f.writelines(table_latex)

        elif self.rc.table_parameter_estimates_format == "csv":
            out = f"{self.config.case_study.output_path}/report_table_parameter_estimates.csv"
            tab.to_csv()


        elif self.rc.table_parameter_estimates_format == "tsv":
            out = f"{self.config.case_study.output_path}/report_table_parameter_estimates.tsv"
            tab.to_csv(out, sep="\t")

        return out
from pymob import SimulationBase, Config
from pymob.solvers.diffrax import JaxSolver
import sys

if __name__ == "__main__":
    config = Config("case_studies/tktd_rna_pulse/scenarios/rna_pulse_3_6c_substance_specific/settings.cfg")
    config.case_study.name = "tktd_rna_pulse"
    config.case_study.package = "case_studies"
    config.import_casestudy_modules(reset_path=True)
    from tktd_rna_pulse.sim import SingleSubstanceSim2

    sim = SingleSubstanceSim2(config)
    sim.set_fixed_parameters(None)
    sim.solver = JaxSolver
    sim.setup()
    sim.generate_artificial_data()

    from tktd_rna_pulse import mod as trpmod

    trpmod.tktd_rna_3_6c

    

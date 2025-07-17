import os
from pymob import Config
from pymob.utils.testing import _compare_config

from tests.fixtures import init_case_study_and_scenario

def test_export_mode_copy(tmp_path):
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_scenario_v2"
    )

    sim.export(directory=os.path.join(tmp_path, "export"), mode="copy")
    exp_config = Config(os.path.join(tmp_path, "export", "settings.cfg"))

    # test if case study configurations are equal except paths in case-study section
    _compare_config(
        config_a=exp_config,
        config_b=exp_config,
    )


def test_export(tmp_path):
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_scenario_v2"
    )

    sim.export(directory=os.path.join(tmp_path, "export"), mode="export")
    exp_config = Config(os.path.join(tmp_path, "export", "settings.cfg"))

    expected_changes_in_case_study_fields = ["data", "scenario_path_override", "output"]

    # test if case study configurations are equal except paths in case-study section
    _compare_config(
        config_a=sim.config,
        config_b=exp_config,
        ignore_fields={"case_study": expected_changes_in_case_study_fields}
    )

    # test case_study paths are equal to the export directory
    for field in expected_changes_in_case_study_fields:
        value = getattr(exp_config.case_study, field)
        assert value == os.path.join(tmp_path, "export")


def test_import(tmp_path):
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_scenario_v2"
    )

    sim.export(directory=os.path.join(tmp_path, "export"))
    cop = sim.from_directory(directory=os.path.join(tmp_path, "export"), mode="import")

    expected_changes_in_case_study_fields = ["data", "scenario_path_override", "output"]

    # test if case study configurations are equal except paths in case-study section
    _compare_config(
        config_a=sim.config,
        config_b=cop.config,
        ignore_fields={"case_study": expected_changes_in_case_study_fields}
    )

    assert sim.observations.equals(cop.observations)

    # test case_study paths are equal to the export directory
    for field in expected_changes_in_case_study_fields:
        value = getattr(cop.config.case_study, field)
        assert value == os.path.join(tmp_path, "export")


def test_import_mode_copy(tmp_path):
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_scenario_v2"
    )

    sim.export(directory=os.path.join(tmp_path, "export"), mode="copy")
    cop = sim.from_directory(directory=os.path.join(tmp_path, "export"), mode="copy")

    # test if case study configurations are equal except paths in case-study section
    _compare_config(
        config_a=sim.config,
        config_b=cop.config,
    )

    # test if observations are equivalent
    assert sim.observations.equals(cop.observations)

def test_copy():
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_scenario_v2"
    )

    cop = sim.copy()

    # test if case study configurations are equal except paths in case-study section
    _compare_config(
        config_a=sim.config,
        config_b=cop.config,
    )

    # test if observations are equivalent
    assert sim.observations.equals(cop.observations)


if __name__ == "__main__":
    test_import(tmp_path="results/debug")
    test_copy()
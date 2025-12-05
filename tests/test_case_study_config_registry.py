import os
import configparser
import pytest
from pymob.sim.config import Config, register_case_study_config
from pydantic import BaseModel, Field
from click.testing import CliRunner

from tests.fixtures import init_case_study_and_scenario

# Dummy case‑study model for testing
class DummySettings(BaseModel):
    foo: int = Field(1, description="dummy int")
    bar: str = Field("baz", description="dummy string")

# Register under a synthetic name
register_case_study_config("dummy_case_study", DummySettings)

@pytest.fixture
def tmp_cfg(tmp_path):
    cfg_path = tmp_path / "settings.cfg"
    parser = configparser.ConfigParser()
    parser["case-study"] = {"name": "dummy_case_study", "scenario": "test"}
    parser["dummy_case_study"] = {"foo": "42", "bar": "hello"}
    parser["dummy_case_study_fail"] = {}
    with open(cfg_path, "w") as f:
        parser.write(f)
    return str(cfg_path)

def test_registry_parses_section(tmp_cfg):
    cfg = Config(tmp_cfg)
    # generic part
    assert cfg.case_study.name == "dummy_case_study"
    # case‑study specific part
    assert cfg.dummy_case_study.foo == 42          # int conversion by Pydantic
    assert cfg.dummy_case_study.bar == "hello"

    # mutate and round‑trip via save()
    cfg.dummy_case_study.foo = 7
    out_path = tmp_cfg + ".out"
    cfg.save(fp=out_path, force=True)

    # reload and verify values persisted
    reloaded = Config(out_path)
    assert reloaded.dummy_case_study.foo == 7
    assert reloaded.dummy_case_study.bar == "hello"

@pytest.fixture
def tmp_cfg_lv(tmp_path):
    cfg_path = tmp_path / "settings.cfg"
    return str(cfg_path)

def test_lotka_volterra_with_case_study_section(tmp_cfg_lv):
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_scenario_v2_with_config_section"
    )

    # in the config file the setting is set to False. The default setting is True
    assert not sim.config.lotka_volterra.test_setting_1

    sim.config.save(fp=tmp_cfg_lv, force=True)

    # reload and verify values persisted
    reloaded = Config(tmp_cfg_lv)
    assert not reloaded.lotka_volterra.test_setting_1


def test_lotka_volterra_without_case_study_section(tmp_cfg_lv):
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_scenario_v2"
    )

    # in the config file the setting is set to False. The default setting is True
    assert sim.config.lotka_volterra.test_setting_1

    # this might be surprising, but the default behavior is to use all registered models
    assert sim.config.dummy_case_study.bar == "baz" # default settings are used 

    sim.config.lotka_volterra.test_setting_1 = False
    sim.config.save(fp=tmp_cfg_lv, force=True)

    # reload and verify values persisted
    reloaded = Config(tmp_cfg_lv)

    # make sure the settings are correctly loaded again (not overwritten by the defaults)
    assert not reloaded.lotka_volterra.test_setting_1



def test_commandline_api_simulate():
    from pymob.simulate import main
    runner = CliRunner()
    
    args = "--case_study=lotka_volterra_case_study "\
        "--scenario=test_scenario_v2_with_config_section "\
        "--package=case_studies"
    result = runner.invoke(main, args.split(" "))

    reloaded = Config("case_studies/lotka_volterra_case_study/scenarios/test_scenario_v2_with_config_section/settings.cfg")
    
    assert not reloaded.lotka_volterra.test_setting_1  # specified setting in file
    assert not reloaded.lotka_volterra.test_setting_2 == "I am Lotka!"  # default setting
    assert reloaded.lotka_volterra.test_setting_2 == "I am Lotka"  # default setting

    if result.exception is not None:
        raise result.exception

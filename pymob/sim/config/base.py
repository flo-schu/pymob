import os
import sys
import configparser
import warnings
import importlib
import importlib.util
from typing import List, Optional, Union, Dict, Literal, Tuple, Any
from types import ModuleType
import click
from pydantic import BaseModel, ConfigDict, Field
from pymob.sim.config.sections import (
    Casestudy,
    Simulation,
    Datastructure,
    Solverbase,
    Jaxsolver,
    Inference,
    Modelparameters,
    Errormodel,
    Multiprocessing,
    Pyabc,
    Pymoo,
    Numpyro,
    Redis,
    Report
)


from pymob.sim.config.casestudy_registry import get_case_study_model, _registry
from pymob.utils.store_file import scenario_file, converters
# this loads at the import of the module
default_path = sys.path.copy()

class Config(BaseModel):
    """
    Configuration manager for *pymob*.

    This class loads a ``settings.cfg`` file, validates it against the
    pydantic models defined above and provides convenient accessors.

    Parameters
    ----------
    config : str or configparser.ConfigParser, optional
        Path to a configuration file or an already parsed ``ConfigParser``.
    """
    model_config = ConfigDict(validate_assignment=True, extra="allow", protected_namespaces=())

    _config: configparser.ConfigParser = configparser.ConfigParser()
    _modules: Dict[str, ModuleType] = {}

    def __init__(
        self,
        config: Optional[Union[str, configparser.ConfigParser]] = None,
    ) -> None:

        _cfg_fp = None
        interp = configparser.ExtendedInterpolation()
        if isinstance(config, str):
            _config = configparser.ConfigParser(
                converters=converters,
                interpolation=interp
            )        
            _config.optionxform = str # type: ignore
            _cfg_file_paths = _config.read(config)
            try:
                _cfg_fp = _cfg_file_paths[0]
            except IndexError:
                raise FileNotFoundError(
                    f"Config file: {config} could not be found."
                ) 
        elif isinstance(config, configparser.ConfigParser):
            _config = config
        else:
            _config = configparser.ConfigParser(
                converters=converters,
                interpolation=interp
            )
            _config.optionxform = str # type: ignore

        # pass arguments to config

        if _cfg_fp is not None: 
            _config.set("case-study", "settings_path", _cfg_fp)
        cfg_dict = {k:dict(s) for k, s in dict(_config).items() if k != "DEFAULT"}
        
        # initalize case_study separately and import modules. This is done here,
        # so that the configuration sections from the registry are discoverable
        # this can only happen once the respective case studies are imported and with them
        # their settings models are defined registered (this of course must be done with the
        # register_case_study_config utility.

        # implementation sketch
        # TODO: Currently, everytime a Config is created (even with default configs)
        #       paths are added to sys path and this is printed. 
        #       by _import_casestudy_modules
        #       The easiest (hotfix) is to surpress these print statemetns when 
        #       the default settings are used.
        #       another (additional) opportunity would be to exploit this, if Config.__init__ is given 
        #       additional arguments (case-study name, scenario and package_location) 
        #       and a flag create_package.
        #       The arguments could alredy be used in the entry points simulate and infer
        #       the flat could be used in conjunction with _import_casestudy_modules 
        #       creating the package location, subdirectories and modules on the fly
        #       and importing them. This would be a nice jumpstarter for starting to work
        #       with case studies
        # if case_study_name is not None:
        #     cfg_dict["case_study"].update({"name": case_study_name})

        # if case_study_scenario is not None:
        #     cfg_dict["case_study"].update({"scenario": case_study_scenario})

        # if case_study_package is not None:
        #     cfg_dict["case_study"].update({"package": case_study_package})

        # if case_study_modules is not None:
        #     cfg_dict["case_study"].update({"modules": case_study_modules})

        _case_study_raw = cfg_dict.get("case-study", {})
        _case_study_instance = Casestudy.model_validate(_case_study_raw)
        _modules = self._import_casestudy_modules(case_study=_case_study_instance, reset_path=True)

        # initialize submodels
        super().__init__(**cfg_dict)

        self._config = _config
        self._modules = _modules

        # Load any case-study-specific configuration after generic sections are parsed
        self._load_case_study_section()

    case_study: Casestudy = Field(default=Casestudy(), alias="case-study")
    simulation: Simulation = Field(default=Simulation())
    data_structure: Datastructure = Field(default=Datastructure(), alias="data-structure") # type:ignore
    solverbase: Solverbase = Field(default=Solverbase())
    jaxsolver: Jaxsolver = Field(default=Jaxsolver(), alias="jax-solver")
    inference: Inference = Field(default=Inference())
    model_parameters: Modelparameters = Field(default=Modelparameters(), alias="model-parameters") #type: ignore
    error_model: Errormodel = Field(default=Errormodel(), alias="error-model") # type: ignore
    multiprocessing: Multiprocessing = Field(default=Multiprocessing())
    inference_pyabc: Pyabc = Field(default=Pyabc(), alias="inference.pyabc")
    inference_pyabc_redis: Redis = Field(default=Redis(), alias="inference.pyabc.redis")
    inference_pymoo: Pymoo = Field(default=Pymoo(), alias="inference.pymoo")
    inference_numpyro: Numpyro = Field(default=Numpyro(), alias="inference.numpyro")
    report: Report = Field(default=Report(), alias="report")
        
    @property
    def input_file_paths(self) -> List[str]:
        """
        List of all input files required for the simulation.

        Returns
        -------
        List[str]
        """
        paths_input_files = []
        for file in self.simulation.input_files:
            fp = scenario_file(file, self.case_study.name, self.case_study.scenario, pkg_dir=self.case_study.package)
            paths_input_files.append(fp)


        file = self.case_study.observations
        if file is None:
            return paths_input_files
        else:
            if not os.path.isabs(file):
                fp = os.path.join(self.case_study.data_path, file)
            else:
                fp = file
            paths_input_files.append(fp)

            return paths_input_files

    def print(self) -> None:
        """Print a summary of the configuration."""
        print("Simulation configuration", end="\n")
        print("========================")
        for section, field_info in self.model_fields.items():
            print(f"{section}({getattr(self, section)})", end="\n") # type: ignore

        print("========================", end="\n")

    def save(self, fp: Optional[str]=None, force: bool=False) -> None:
        """
        Save the configuration to a ``settings.cfg`` file.

        Uses serializers defined at the top, which parse the options to str
        so they can be processed by configfile. 

        In case the model configuration should be stored to a json file use
        something like `json.dumps(self.model_dump())`, because the build in
        function, is somewhat disabled by the listparsers which are needed for
        configfile lists.

        Parameters
        ----------
        fp : Optional[str]
            File path to write the settings file to. If ``None`` the default
            location derived from the case study is used.
        force : bool, optional
            Overwrite without prompting. Default is ``False``.
        """
        settings = self.model_dump(
            by_alias=True, 
            mode="json", 
            exclude_none=True,
            exclude={"case_study": {"output_path", "data_path", "root", "init_root", "default_settings_path"}}
        )
        self._config.update(**settings)

        if fp is None:
            file_path = self.case_study.default_settings_path
        else:
            file_path = os.path.abspath(fp)

        write = True
        if os.path.exists(file_path) and not force:
            ui_overwrite = input("Settings file already exists. Overwrite? [y/N]")
            write = True if ui_overwrite == "y" else False
        else:
            # create a directory for the new scenario file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if write:
            with open(file_path, "w") as f:
                self._config.write(f)

    def create_directory(self, directory: Literal["results", "scenario"], force: bool=False) -> None:
        """
        Create a results or scenario directory if it does not exist.

        Parameters
        ----------
        directory : Literal["results", "scenario"]
            Which directory to create.
        force : bool, optional
            If ``True`` create without prompting.
        """
        if directory == "results":
            p = os.path.abspath(self.case_study.output_path)
        elif directory == "scenario":
            p = os.path.abspath(self.case_study.scenario_path)
        else:
            raise NotImplementedError(
                f"{directory.capitalize()} is not an expected directory in the " +
                "case study logic. Use one of 'results' and 'scenario'."
            )

        if os.path.exists(p):
            print(f"{directory.capitalize()} directory exists at '{p}'.")
            return
        else:
            if not force:
                answer = input(f"Create {directory} directory at '{p}'? [Y/n]")
            else:
                answer = "y"

            if answer.lower() == "y" or answer.lower() == "":
                os.makedirs(p)
                print(f"{directory.capitalize()} directory created at '{p}'.")
            else:
                print(f"No {directory} directory created.")

    def import_casestudy_modules(self, reset_path: bool=False) -> None:
        """
        Import all modules of the current case study.

        this script handles the import of a case study without the typical 
        __init__.py file. It iterates over all .py files in the root directory
        of the case study (typically: sim, mod, stats, plot, data, prior)
        and imports them with import_module(...)

        Parameters
        ----------
        reset_path : bool, optional
            Reset ``sys.path`` before importing. Default is ``False``.

        """
        modules = self._import_casestudy_modules(
            case_study=self.case_study, 
            reset_path=reset_path
        )
        
        self._modules.update(modules)

    @staticmethod
    def _import_casestudy_modules(case_study: Casestudy, reset_path: bool=False) -> Dict[str, ModuleType]:
        """
        Import modules for a given case study.

        Parameters
        ----------
        case_study : Casestudy
            The case-study configuration.
        reset_path : bool, optional
            Whether to reset ``sys.path`` before importing.

        Returns
        -------
        Dict[str, ModuleType]
            Mapping of module name to imported module.
        """
        _modules = {}


        # reset the path to avoid importing modules form case-studies used
        # before in the same session
        if reset_path:
            # default path needs be copied, otherwise it will be updated
            # when setting sys.path
            sys.path = default_path.copy()


        # potential BUG: This is not safe. It is not guaranteed that the 
        # case study has the same name as the package. But it might be in the future
        package = case_study.name

        if "-" in package or " " in package:
            warnings.warn(
                f"Case-study contained {package} contained unallowed "+
                "characters: ['-', ' ']. "+
                "The characters will be replaced with underscores ('_') for "+
                "importing the package modules. " +
                "In the future, the name of the case study should be the same as " +
                "as the package where the modules are located. This name must not " +
                "contain hyphens ('-') or whitespace characters (' '),",
                category=UserWarning
            )
            _package = package.replace("-", "_").replace(" ", "_")
        else:
            _package = package

        spec = importlib.util.find_spec(_package)
        if spec is not None:
            for module in case_study.modules:
                try:
                    # TODO: Consider importing modules as a nested dictionary 
                    # with the indexing key being the package. The package
                    # cannot be derived from the class, if a method, that is 
                    # executed on a lower level case-study, should target that 
                    # a module belonging to the same package, because if the
                    # object is used, it would resolve to the package of the
                    # higher level case-study
                    m = importlib.import_module(f"{_package}.{module}")
                    _modules.update({module: m})
                except ModuleNotFoundError:
                    warnings.warn(
                        f"Module {module}.py not found in {_package}." +
                        "Missing modules can lead to unexpected behavior. " +
                        f"Does your case study have a {module}.py file? " +
                        "It should have the line `from PARENT_CASE_STUDY." +
                        f"{module} import *` to import all objects from " +
                        "the parent case study."
                    )
            return _modules

        # append relevant paths to sys
        package = os.path.join(
            case_study.root, 
            case_study.package
        )
        if package not in sys.path:
            sys.path.insert(0, package)
            print(f"Inserted '{package}' in PATH at index=0")
    
        case_study_path = os.path.join(
            case_study.root, 
            case_study.package,
            case_study.name,
            # Account for package architecture 
            case_study.name
        )
        if case_study_path not in sys.path:
            sys.path.insert(0, case_study_path)
            print(f"Inserted '{case_study_path}' in PATH at index=0")

        for module in case_study.modules:
            # remove modules of a different case study that might have been
            # loaded in the same session.
            if module in sys.modules:
                _ = sys.modules.pop(module)

        for module in case_study.modules:
            try:
                m = importlib.import_module(module, package=case_study_path)
                _modules.update({module: m})
            except ModuleNotFoundError:
                warnings.warn(
                    f"Module {module}.py not found in {case_study_path}." +
                    "Missing modules can lead to unexpected behavior." +
                    "If a module is not imported, you can specify it in the " +
                    "Config 'config.case_study.modules = [...]'"
                )

        return _modules

    def import_simulation_from_case_study(self) -> Any:
        """
        Retrieve the ``Simulation`` class defined in the case-study.

        Returns
        -------
        Any
            The ``Simulation`` class object.

        Raises
        ------
        ImportError
            If the class cannot be found.
        """
        try:
            Simulation = getattr(self._modules["sim"], self.case_study.simulation)
        except Exception as e:
            raise ImportError(
                f"Simulation class '{self.case_study.simulation}' " +
                "could not be found. Make sure the simulaton option is spelled " +
                "correctly or specify a class that exists in sim.py" +
                "If you are using pymob to work on different case-studies in " +
                "the same session, make sure to reset the path by " +
                "using `import_casestudy_modules(reset_path=True)`" +
                f"\nOriginal exception: {e}"
            )
        
        return Simulation

    def set_option(self, section: str, option: str, value: str) -> None:
        """
        Set a configuration option.

        Parameters
        ----------
        section : str
            Name of the configuration section (e.g. ``simulation``).
        option : str
            Option name within the section.
        value : str
            New value as a string; will be parsed according to the section's
            type definitions.
        """
        sect = getattr(self, section)
        if isinstance(sect, Modelparameters):
            if (value == "" or value == "None") and option in sect.all:
                sect.remove(option)
            elif (value == "" or value == "None") and option not in sect.all:
                pass
            else:
                sect[option] = value
        else:
            sect[option] = value

    def _load_case_study_section(self) -> None:
        """
        Parse a registered case-study-specific INI section and expose it as an attribute.
        """
        # Ensure the model has extra sections
        if self.model_extra is None:
            return

        # parse the config sections if they are defined in the settings.cfg
        # this overwrites the default definitions
        for section, options in self.model_extra.items():
            # get the model from the registry
            model_cls = get_case_study_model(section)

            if model_cls is None:
                warnings.warn(
                    f"Config section {section} could not be parsed. " +
                    "You need to register a the Config Model like that:\n\n" +
                    ">>> from pymob.sim.casestudy_registry import register_case_study_config\n" +
                    ">>> from pymob.sim.config import PymobModel\n" +
                    f">>> class {section.capitalize()}Config(PymobModel):\n" +
                    ">>>     option_a: float = 1.0\n" +
                    f">>> register_case_study_config({section.capitalize()}Config)\n"
                )

            else:
                # Pydantic parses/validates the raw strings
                instance = model_cls.model_validate(options, strict=False)
                # Store on the Config object – attribute name equals the case-study name
                self.model_extra.update({section: instance})

        # add default sections in case the sections are not defined in the settings.cfg
        for section, model_cls in _registry.items():
            if section not in self.model_extra:
                default_instance = model_cls.model_validate({}, strict=False)
                self.model_extra.update({section: default_instance})
            

@click.command
@click.option("--file", "-f", type=str, nargs=1, help="Path to the config file (usually in scenario/.../settings.cfg)")
@click.option("--options", "-o", type=str, multiple=True, help="The option or options to configure. Combine sections and option like this 'simulation.seed=1' options that have spaces need to be wraped in quotes")
def configure(file: str, options: Tuple[str, ...]) -> None:
    """
    Command-line entry point to modify a configuration file.

    Parameters
    ----------
    file : str
        Path to the configuration file.
    options : Tuple[str, ...]
        List of ``section.option=value`` strings.

    Examples
    --------
    >>> # Change the seed of the simulation
    >>> configure("-f", "scenario/settings.cfg", "-o", "simulation.seed=42")
    """
    config = Config(file)
    for opt in options:
        key, val = opt.split("=", 1)

        section, option = key.strip(" ").rsplit(".", 1)

        section = section.replace("-","_").replace(".","_")

        if section == "jax_solver":
            section = "jaxsolver"

        value = val.strip(" ")
        config.set_option(section, option, value)

    config.save(file, force=True)

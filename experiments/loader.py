"""A simple plugin loader."""
import importlib
from pkgutil import iter_modules
from typing import cast
from typing import Protocol

from chemex.experiments import catalog


class ExperimentModuleInterface(Protocol):
    """Represents an experiment module interface.
    An experiment module plugin has a single register function."""

    @staticmethod
    def register() -> None:
        """Register the necessary items related to the experiment."""
        ...


def import_module(name: str) -> ExperimentModuleInterface:
    """Imports a module given a name."""
    return cast(ExperimentModuleInterface, importlib.import_module(name))


def register_experiments() -> None:
    """Loads the plugins defined in the plugins list."""
    for package in iter_modules(catalog.__path__):
        full_package_name = f"{catalog.__name__}.{package.name}"
        experiment_module = import_module(f"{full_package_name}.experiment")
        experiment_module.register()

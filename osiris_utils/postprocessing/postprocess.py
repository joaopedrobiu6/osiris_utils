from __future__ import annotations

from typing import Any

from ..data.diagnostic import Diagnostic

__all__ = ["PostProcess"]


class PostProcess:
    """
    Base class for post-processing wrappers (FFT_Simulation, Derivative_Simulation, etc).

    It wraps a Simulation-like object (Simulation or another PostProcess) and exposes:
      - .species
      - .loaded_diagnostics
      - .add_diagnostic()

    Subclasses implement __getitem__ to return processed diagnostics.
    """

    def __init__(self, name: str, simulation: Any):
        self._name = name
        self._simulation = simulation

        # Simulation-like capability checks (supports chaining)
        if not hasattr(simulation, "__getitem__"):
            raise TypeError("Wrapped object must support __getitem__ like Simulation.")
        if not (hasattr(simulation, "_species") or hasattr(simulation, "species")):
            raise TypeError("Wrapped object must have _species or species attribute.")

        # normalize species
        self._species = getattr(simulation, "_species", simulation.species)

    @property
    def species(self) -> list:
        return self._species

    @property
    def loaded_diagnostics(self) -> dict:
        """
        Delegate to wrapped simulation if it has loaded_diagnostics.
        Otherwise return empty dict.
        """
        return getattr(self._simulation, "loaded_diagnostics", {})

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """
        Delegate add_diagnostic downwards.
        This lets you do: sim_fft.add_diagnostic(...)
        """
        if not hasattr(self._simulation, "add_diagnostic"):
            raise AttributeError("Wrapped simulation does not support add_diagnostic().")
        return self._simulation.add_diagnostic(diagnostic, name=name)

    def delete_all_diagnostics(self) -> None:
        if hasattr(self._simulation, "delete_all_diagnostics"):
            self._simulation.delete_all_diagnostics()

    def delete_diagnostic(self, key: str) -> None:
        if hasattr(self._simulation, "delete_diagnostic"):
            self._simulation.delete_diagnostic(key)

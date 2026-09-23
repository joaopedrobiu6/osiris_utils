from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..data.simulation import _register

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..data.diagnostic import Diagnostic

__all__ = ["PostProcess"]

logger = logging.getLogger(__name__)


class PostProcess:
    """
    Base class for post-processing views (Derivative_Simulation, MFT_Simulation, ...).

    A view wraps a Simulation — or another view, so they chain — and hands out
    processed diagnostics: ``view[key]`` is :meth:`_process` applied to the
    wrapped simulation's *key*, built on first access and kept.
    ``view[species]`` is the same view over that species' quantities.  A
    subclass therefore implements one method, :meth:`_process`.

    Parameters
    ----------
    name : str
        Label used in messages.
    simulation :
        What to wrap: anything with ``__getitem__`` and ``species``.
    species : sequence of str, optional
        Keys that open a species view.  Default: ``simulation.species``; pass
        ``()`` for a view with no species quantities (e.g. field centering).
    """

    def __init__(self, name: str, simulation: Any, species: Sequence[str] | None = None) -> None:
        if not hasattr(simulation, "__getitem__"):
            raise TypeError("Wrapped object must support __getitem__ like Simulation.")
        if species is None and not hasattr(simulation, "species"):
            raise TypeError("Wrapped object must have a species attribute.")
        self._name = name
        self._simulation = simulation
        self._species = list(simulation.species if species is None else species)
        self._built: dict[str, Any] = {}  # processed diagnostics and species views, built on first access
        self._custom: dict[str, Diagnostic] = {}  # registered with add_diagnostic, handed back as they are

    def _process(self, container: Any, key: str) -> Diagnostic:
        """The processed diagnostic for *key* of *container*: the wrapped simulation, or one of its species."""
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        if key in self._custom:
            return self._custom[key]
        if key not in self._built:
            self._built[key] = _SpeciesView(self, key) if key in self._species else self._process(self._simulation, key)
        return self._built[key]

    @property
    def species(self) -> list:
        return self._species

    @property
    def loaded_diagnostics(self) -> dict[str, Any]:
        """Every diagnostic this view has built or been given."""
        return {**self._built, **self._custom}

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """Register *diagnostic* on this view; ``view[name]`` hands it back unprocessed.

        Use it for anything built *from* the view (products, derivatives,
        e_vlasov): the processing already happened in its leaves.  It is stored
        here rather than on the wrapped simulation because its value depends on
        the view that built it — two filtered views with different filters must
        not pick up each other's composites.
        """
        return _register(self._custom, diagnostic, name)

    def delete_diagnostic(self, key: str) -> None:
        """Forget *key* on this view (the wrapped simulation is untouched)."""
        found = self._custom.pop(key, None) is not None
        found = self._built.pop(key, None) is not None or found
        if not found:
            logger.warning("Diagnostic '%s' not found in %s.", key, self._name)

    def delete_all_diagnostics(self) -> None:
        """Forget everything this view built or was given."""
        self._built = {}
        self._custom = {}


class _SpeciesView(PostProcess):
    """``view[species]``: the parent view's processing, applied to that species' quantities."""

    def __init__(self, parent: PostProcess, species: str) -> None:
        super().__init__(f"{parent._name}[{species}]", parent._simulation[species], species=())
        self._parent = parent

    def _process(self, container: Any, key: str) -> Diagnostic:
        return self._parent._process(container, key)

    @property
    def species(self) -> Any:
        """The species itself, as the wrapped species handler reports it."""
        return getattr(self._simulation, "species", None)

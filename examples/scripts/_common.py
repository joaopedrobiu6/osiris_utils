"""Shared plumbing for the example scripts: arguments, banners, plots.

Every example accepts the same two options::

    python examples/scripts/<script>.py                 # synthetic run (default)
    python examples/scripts/<script>.py --sim RUN/os-stdin   # your own run
    python examples/scripts/<script>.py --plot           # also write PNGs

so the same code can be pointed at real OSIRIS output without editing it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# The examples are run as scripts, not installed, so make the sibling modules
# (synthetic_run) importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_run import default_run  # noqa: E402

__all__ = ["parse_args", "deck_path", "load_simulation", "section", "show", "savefig", "PLOT_DIR"]

PLOT_DIR = Path(__file__).resolve().parent / "output"


def parse_args(description: str, extra: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    """Standard example CLI: ``--sim``, ``--plot``, ``--outdir``."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", default=None, help="path to an OSIRIS input deck (default: synthetic run)")
    parser.add_argument("--plot", action="store_true", help="save figures to --outdir")
    parser.add_argument("--outdir", default=str(PLOT_DIR), help=f"where figures and files go (default: {PLOT_DIR})")
    parser.add_argument("--log", action="store_true", help="keep the package's INFO logging (off by default)")
    for flags, kwargs in extra or []:
        parser.add_argument(flags, **kwargs)
    args = parser.parse_args()
    args.outdir = Path(args.outdir)

    # osiris_utils logs at INFO by default (diagnostic.py calls basicConfig).
    # Useful in a real analysis, noise in an example.
    if not args.log:
        logging.getLogger("osiris_utils").setLevel(logging.WARNING)
    return args


def deck_path(args: argparse.Namespace, **synthetic_kwargs) -> Path:
    """The input deck an example works on: the user's run, or the synthetic one."""
    return Path(args.sim) if args.sim else default_run(**synthetic_kwargs)


def load_simulation(args: argparse.Namespace, **synthetic_kwargs):
    """``Simulation`` built from :func:`deck_path`."""
    import osiris_utils as ou

    deck = deck_path(args, **synthetic_kwargs)
    print(f"Simulation deck : {deck}")
    return ou.Simulation(str(deck))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(label: str, value) -> None:
    """One aligned ``label : value`` line, arrays summarised rather than dumped."""
    import numpy as np

    if isinstance(value, np.ndarray) and value.size > 6:
        value = f"ndarray{value.shape} {value.dtype}  min={value.min():.4g}  max={value.max():.4g}"
    print(f"  {label:<34} {value}")


def savefig(fig, name: str, args) -> None:
    """Write *fig* into ``--outdir`` (no-op unless ``--plot`` was given)."""
    if not args.plot:
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  [figure] {path}")

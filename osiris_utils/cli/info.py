"""Info command - display metadata about OSIRIS files and simulations."""

import argparse
import sys
from pathlib import Path

import osiris_utils as ou


def register_parser(subparsers) -> None:
    """Register the 'info' subcommand parser."""
    parser = subparsers.add_parser(
        "info",
        help="Display metadata about OSIRIS files or simulations",
        description="Show detailed information about OSIRIS simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  utils info path/to/input.deck           # Show all simulation info
  utils info path/to/file.h5              # Show single file info
  utils info path/to/input.deck --brief   # Show summary only
        """,
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to OSIRIS input deck or HDF5 file",
    )

    parser.add_argument(
        "--brief",
        action="store_true",
        help="Show brief summary only",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the info command."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path '{path}' does not exist", file=sys.stderr)
        return 1

    # Determine if path is a file or directory
    if path.is_file():
        if path.name in ["os-stdin", "input.deck", "deck.in"] or path.suffix == ".deck":
            return show_simulation_info(path, args.brief, args.verbose)
        return show_file_info(path, args.brief)
    elif path.is_dir():
        # Fallback for folder access: try to find the deck automatically
        for candidate in ["os-stdin", "input.deck", "deck.in"]:
            deck_path = path / candidate
            if deck_path.exists():
                return show_simulation_info(deck_path, args.brief, args.verbose)

        print(f"Error: No input deck found in {path}", file=sys.stderr)
        return 1
    else:
        print(f"Error: '{path}' is not a file or directory", file=sys.stderr)
        return 1


def show_file_info(filepath: Path, brief: bool = False) -> int:
    """Display information about a single OSIRIS file."""
    try:
        # Try to open as grid file
        data = ou.OsirisGridFile(str(filepath))

        print(f"File: {filepath.name}")
        print(f"Type: {data.type}")
        print(f"Name: {data.name}")
        print(f"Dimensions: {data.dim}D")

        if not brief:
            print("\nGrid Information:")
            print(f"  nx: {data.nx}")
            print(f"  dx: {data.dx}")
            print(f"  Grid range: {[tuple(ax['axis']) for ax in data.axis]}")

            print("\nTime Information:")
            print(f"  Time: {data.time}")
            print(f"  dt: {data.dt}")
            print(f"  Iteration: {data.iter}")

            print("\nData Information:")
            print(f"  Shape: {data.data.shape}")
            print(f"  Units: {data.units}")
            print(f"  Label: {data.label}")

        return 0

    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1


def show_simulation_info(deck_path: Path, brief: bool = False, verbose: bool = False) -> int:
    """Display information about an OSIRIS simulation."""
    simpath = deck_path.parent
    input_deck = deck_path

    try:
        # Load simulation
        sim = ou.Simulation(str(input_deck))

        print(f"Simulation: {simpath.name}")
        print(f"Input Deck: {input_deck.name}")

        if not brief:
            print("\nSpecies:")
            for species in sim.species:
                print(f"  - {species}")

            # Scan for available diagnostics
            print("\nAvailable Diagnostics:")
            ms_path = simpath / "MS"
            if ms_path.exists():
                # Check for fields
                fld_path = ms_path / "FLD"
                if fld_path.exists():
                    print("  Fields:")
                    for item in sorted(fld_path.iterdir()):
                        if item.is_dir():
                            # Count files in diagnostic
                            n_files = len(list(item.glob("*.h5")))
                            print(f"    - {item.name} ({n_files} timesteps)")

                # Check for density/current
                for diag_type in ["DENSITY", "CURRENT"]:
                    diag_path = ms_path / diag_type
                    if diag_path.exists():
                        print(f"  {diag_type.title()}:")
                        for species_dir in sorted(diag_path.iterdir()):
                            if species_dir.is_dir():
                                for qty_dir in sorted(species_dir.iterdir()):
                                    if qty_dir.is_dir():
                                        n_files = len(list(qty_dir.glob("*.h5")))
                                        print(f"    - {species_dir.name}/{qty_dir.name} ({n_files} timesteps)")

        return 0

    except Exception as e:
        print(f"Error reading simulation: {e}", file=sys.stderr)
        if verbose:
            raise
        return 1


# Make args available in scope for error handling
args = None

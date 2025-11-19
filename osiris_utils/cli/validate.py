"""Validate command - check OSIRIS file and simulation integrity."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

import osiris_utils as ou


def register_parser(subparsers) -> None:
    """Register the 'validate' subcommand parser."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate OSIRIS files and simulations",
        description="Check integrity of OSIRIS data files and simulation structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  osiris validate path/to/simulation      # Check entire simulation
  osiris validate file.h5                 # Check single file
  osiris validate sim --check-missing     # Check for missing timesteps
        """,
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to OSIRIS simulation directory or HDF5 file",
    )

    parser.add_argument(
        "--check-missing",
        action="store_true",
        help="Check for missing timesteps in diagnostics",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings, not just errors",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path '{path}' does not exist", file=sys.stderr)
        return 1

    errors = 0
    warnings = 0

    try:
        if path.is_file():
            e, w = validate_file(path)
            errors += e
            warnings += w
        elif path.is_dir():
            e, w = validate_simulation(path, args.check_missing)
            errors += e
            warnings += w
        else:
            print(f"Error: '{path}' is not a file or directory", file=sys.stderr)
            return 1

        # Summary
        separator = '=' * 60
        print(f"\n{separator}")
        print("Validation Summary:")
        print(f"  Errors: {errors}")
        print(f"  Warnings: {warnings}")

        if errors > 0:
            print("\nValidation FAILED")
            return 1
        elif warnings > 0 and args.strict:
            print("\nValidation FAILED (strict mode)")
            return 1
        else:
            print("\nValidation PASSED")
            return 0

    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        return 1


def validate_file(filepath: Path) -> tuple[int, int]:
    """
    Validate a single HDF5 file.

    Returns
    -------
    tuple[int, int]
        (error_count, warning_count)
    """
    errors = 0
    warnings = 0

    print(f"Validating file: {filepath}")

    # Check if file can be opened
    try:
        with h5py.File(filepath, 'r') as f:
            print("File is a valid HDF5 file")

            # Check for required datasets
            if 'AXIS' not in f:
                print("Warning: Missing AXIS dataset")
                warnings += 1

            # Try to load with osiris_utils
            try:
                data = ou.OsirisGridFile(str(filepath))
                print("File loads successfully with osiris_utils")

                # Check data integrity
                if data.data.size == 0:
                    print("Error: Data array is empty")
                    errors += 1
                else:
                    print(f"Data array is non-empty (shape: {data.data.shape})")

                # Check for NaN or Inf
                if not np.isfinite(data.data).all():
                    print("Warning: Data contains NaN or Inf values")
                    warnings += 1
                else:
                    print("Data contains only finite values")

            except Exception as e:
                print(f"Error: Cannot load with osiris_utils: {e}")
                errors += 1

    except OSError as e:
        print(f"Error: Cannot open file: {e}")
        errors += 1

    return errors, warnings


def validate_simulation(simpath: Path, check_missing: bool = False) -> tuple[int, int]:
    """
    Validate an entire simulation directory.

    Returns
    -------
    tuple[int, int]
        (error_count, warning_count)
    """
    errors = 0
    warnings = 0

    print(f"Validating simulation: {simpath}")

    # Check for input deck
    input_deck = None
    for candidate in ["os-stdin", "input.deck", "deck.in"]:
        deck_path = simpath / candidate
        if deck_path.exists():
            input_deck = deck_path
            break

    if input_deck is None:
        print("Error: No input deck found")
        errors += 1
        return errors, warnings
    else:
        print(f"Found input deck: {input_deck.name}")

    # Try to load simulation
    try:
        sim = ou.Simulation(str(input_deck))
        print("Simulation loads successfully")
        print(f"Found {len(sim.species)} species: {', '.join(sim.species)}")
    except Exception as e:
        print(f"Error: Cannot load simulation: {e}")
        errors += 1
        return errors, warnings

    # Check MS directory
    ms_path = simpath / "MS"
    if not ms_path.exists():
        print("Warning: MS directory not found")
        warnings += 1
        return errors, warnings
    else:
        print("Found MS directory")

    # Check diagnostic directories
    diag_types = ["FLD", "DENSITY", "CURRENT", "PHA"]
    for diag_type in diag_types:
        diag_path = ms_path / diag_type
        if diag_path.exists():
            e, w = validate_diagnostic_dir(diag_path, check_missing)
            errors += e
            warnings += w

    return errors, warnings


def validate_diagnostic_dir(diag_path: Path, check_missing: bool) -> tuple[int, int]:
    """Validate a diagnostic directory."""
    errors = 0
    warnings = 0

    print(f"\n  Checking {diag_path.name}:")

    # Recursively find all h5 files
    for subdir in diag_path.rglob("*"):
        if subdir.is_dir():
            h5_files = sorted(list(subdir.glob("*.h5")))
            if h5_files:
                rel_path = subdir.relative_to(diag_path)
                print(f"    {rel_path}: {len(h5_files)} files")

                if check_missing:
                    # Check for sequential numbering
                    actual_indices = set()

                    for f in h5_files:
                        # Extract iteration number from filename
                        try:
                            # Typical format: name-123456.h5
                            iter_str = f.stem.split('-')[-1]
                            actual_indices.add(int(iter_str))
                        except (ValueError, IndexError):
                            pass

                    if len(actual_indices) > 0:
                        min_idx = min(actual_indices)
                        max_idx = max(actual_indices)
                        expected_sequential = set(range(min_idx, max_idx + 1))
                        missing = expected_sequential - actual_indices

                        if missing:
                            print(f"Warning: Missing iterations: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
                            warnings += 1

    return errors, warnings

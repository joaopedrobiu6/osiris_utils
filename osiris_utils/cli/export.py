"""Export command - convert OSIRIS data to different formats."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import osiris_utils as ou


def register_parser(subparsers) -> None:
    """Register the 'export' subcommand parser."""
    parser = subparsers.add_parser(
        "export",
        help="Export OSIRIS data to CSV, JSON, or NumPy formats",
        description="Convert OSIRIS HDF5 data to common formats for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  osiris export file.h5 --format csv --output data.csv
  osiris export sim/MS/FLD/e3 --format npy --output e3_data.npy
  osiris export file.h5 --format json --output data.json
        """,
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to OSIRIS file or diagnostic directory",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["csv", "json", "npy"],
        default="csv",
        help="Output format (default: csv)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )

    parser.add_argument(
        "-t",
        "--timestep",
        type=int,
        default=None,
        help="Specific timestep index to export (default: export all)",
    )

    parser.add_argument(
        "--no-coords",
        action="store_true",
        help="Exclude coordinate information (data only)",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the export command."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path '{path}' does not exist", file=sys.stderr)
        return 1

    try:
        # Load data
        if path.is_file():
            data_obj = ou.OsirisGridFile(str(path))
            export_single_file(data_obj, args)
        elif path.is_dir():
            # Assume it's a diagnostic directory
            export_diagnostic_dir(path, args)
        else:
            print(f"Error: '{path}' is not a file or directory", file=sys.stderr)
            return 1

        print(f"Exported to: {args.output}")
        return 0

    except Exception as e:
        print(f"Error exporting data: {e}", file=sys.stderr)
        return 1


def export_single_file(data_obj, args: argparse.Namespace) -> None:
    """Export a single OSIRIS file."""
    data = data_obj.data
    output_path = Path(args.output)

    if args.format == "csv":
        export_to_csv(data, data_obj, output_path, args.no_coords)
    elif args.format == "json":
        export_to_json(data, data_obj, output_path, args.no_coords)
    elif args.format == "npy":
        np.save(output_path, data)


def export_diagnostic_dir(diag_path: Path, args: argparse.Namespace) -> None:
    """Export all timesteps from a diagnostic directory."""
    # Find all h5 files
    h5_files = sorted(list(diag_path.glob("*.h5")))

    if not h5_files:
        raise ValueError(f"No HDF5 files found in {diag_path}")

    if args.timestep is not None:
        # Export specific timestep
        if args.timestep >= len(h5_files):
            raise ValueError(f"Timestep {args.timestep} out of range (0-{len(h5_files) - 1})")
        data_obj = ou.OsirisGridFile(str(h5_files[args.timestep]))
        export_single_file(data_obj, args)
    else:
        # Export all timesteps
        if args.format == "npy":
            # Stack all data into single array
            all_data = []
            for f in h5_files:
                data_obj = ou.OsirisGridFile(str(f))
                all_data.append(data_obj.data)
            stacked = np.array(all_data)
            np.save(args.output, stacked)
        else:
            # For CSV/JSON, export each timestep separately or create multi-index
            export_multi_timestep(h5_files, args)


def export_to_csv(data: np.ndarray, data_obj, output_path: Path, no_coords: bool) -> None:
    """Export data to CSV format."""
    if data.ndim == 1:
        # 1D data
        if no_coords:
            df = pd.DataFrame({data_obj.name: data})
        else:
            df = pd.DataFrame({"x": data_obj.x[0], data_obj.name: data})
    elif data.ndim == 2:
        # 2D data - flatten with coordinates
        if no_coords:
            df = pd.DataFrame(data)
        else:
            x = data_obj.x[0]
            y = data_obj.x[1]
            xx, yy = np.meshgrid(x, y, indexing='ij')
            df = pd.DataFrame({'x': xx.flatten(), 'y': yy.flatten(), data_obj.name: data.flatten()})
    else:
        # 3D+ data - just flatten
        df = pd.DataFrame(data.flatten(), columns=[data_obj.name])

    df.to_csv(output_path, index=False)


def export_to_json(data: np.ndarray, data_obj, output_path: Path, no_coords: bool) -> None:
    """Export data to JSON format."""
    output = {
        "name": data_obj.name,
        "type": data_obj.type,
        "units": data_obj.units,
        "time": data_obj.time,
        "iteration": data_obj.iter,
        "data": data.tolist(),
    }

    if not no_coords:
        output["grid"] = {f"x{i + 1}": coord.tolist() for i, coord in enumerate(data_obj.x)}
        output["axis_info"] = data_obj.axis

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def export_multi_timestep(h5_files: list, args: argparse.Namespace) -> None:
    """Export multiple timesteps to CSV/JSON."""
    if args.format == "csv":
        # Create multi-index CSV
        all_dfs = []
        for i, f in enumerate(h5_files):
            data_obj = ou.OsirisGridFile(str(f))
            df = pd.DataFrame({'timestep': i, 'time': data_obj.time, 'data': data_obj.data.flatten()})
            all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(args.output, index=False)
    elif args.format == "json":
        # Create JSON array
        all_data = []
        for f in h5_files:
            data_obj = ou.OsirisGridFile(str(f))
            all_data.append({"time": data_obj.time, "iteration": data_obj.iter, "data": data_obj.data.tolist()})
        with open(args.output, 'w') as outf:
            json.dump(all_data, outf, indent=2)

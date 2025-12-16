"""Command-line interface for osiris_utils."""

import argparse
import sys
from typing import List, Optional

import osiris_utils

__version__ = osiris_utils.__version__


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the osiris CLI.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="osiris",
        description="Command-line tools for OSIRIS plasma simulation data analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  osiris info path/to/simulation          # Show simulation metadata
  osiris export file.h5 --format csv      # Export data to CSV
  osiris plot file.h5 --save plot.png     # Create quick plot
  osiris validate path/to/simulation      # Check file integrity

For help on a specific command:
  osiris <command> --help
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"osiris_utils {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
        required=True,
    )

    # Import command modules
    from . import export, info, plot, validate

    # Register each command's parser
    info.register_parser(subparsers)
    export.register_parser(subparsers)
    plot.register_parser(subparsers)
    validate.register_parser(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)

    # Execute the appropriate command
    try:
        return args.func(args)
    except Exception as e:
        if args.verbose:
            raise
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

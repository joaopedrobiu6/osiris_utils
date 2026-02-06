"""Command-line interface for osiris_utils."""

import argparse
import sys

import osiris_utils

from . import export, info, plot, validate

__version__ = osiris_utils.__version__


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the osiris CLI.

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
        prog="utils",
        description="Command-line tools for OSIRIS plasma simulation data analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  utils info path/to/input.deck           # Show simulation metadata
  utils export file.h5 --format csv       # Export data to CSV
  utils plot file.h5 --save plot.png      # Create quick plot
  utils validate path/to/input.deck       # Check file integrity

For help on a specific command:
  utils <command> --help
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
    except Exception:
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Plot command - quick visualization of OSIRIS data."""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import osiris_utils as ou

# Use non-interactive backend by default for CLI
matplotlib.use('Agg')


def register_parser(subparsers) -> None:
    """Register the 'plot' subcommand parser."""
    parser = subparsers.add_parser(
        "plot",
        help="Create quick plots from OSIRIS data",
        description="Generate publication-quality plots from OSIRIS simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  osiris plot file.h5 --save plot.png
  osiris plot file.h5 --save plot.png --title "Ez Field"
  osiris plot file.h5 --save plot.png --cmap viridis
  osiris plot file.h5 --display  # Show interactive plot
        """,
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to OSIRIS HDF5 file",
    )

    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Save plot to file (e.g., plot.png)",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display plot interactively (requires X server)",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (default: auto-generated)",
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="RdBu_r",
        help="Colormap for 2D plots (default: RdBu_r)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots (default: 150)",
    )

    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale for color/y-axis",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the plot command."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: File '{path}' does not exist", file=sys.stderr)
        return 1

    if not path.is_file():
        print(f"Error: '{path}' is not a file", file=sys.stderr)
        return 1

    if not args.save and not args.display:
        print("Error: Must specify --save or --display", file=sys.stderr)
        return 1

    try:
        # Switch to interactive backend if displaying
        if args.display:
            matplotlib.use('TkAgg')

        # Load data
        data_obj = ou.OsirisGridFile(str(path))

        # Create plot based on dimensionality
        if data_obj.dim == 1:
            create_1d_plot(data_obj, args)
        elif data_obj.dim == 2:
            create_2d_plot(data_obj, args)
        else:
            print(f"Error: {data_obj.dim}D plotting not supported yet", file=sys.stderr)
            return 1

        # Save or display
        if args.save:
            plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
            print(f"Plot saved to: {args.save}")

        if args.display:
            plt.show()

        return 0

    except Exception as e:
        print(f"Error creating plot: {e}", file=sys.stderr)
        return 1


def create_1d_plot(data_obj, args: argparse.Namespace) -> None:
    """Create a 1D line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # For 1D data, x is the coordinate array directly
    x = data_obj.x
    data = data_obj.data

    ax.plot(x, data, linewidth=2)

    # Labels
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"${data_obj.label}$ at t = {data_obj.time}")

    ax.set_xlabel(data_obj.axis[0].get('plot_label', f'x [{data_obj.axis[0]["units"]}]'))
    ax.set_ylabel(f"${data_obj.label}$ [{data_obj.units}]")

    if args.log_scale:
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def create_2d_plot(data_obj, args: argparse.Namespace) -> None:
    """Create a 2D heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    x = data_obj.x[0]
    y = data_obj.x[1]
    data = data_obj.data

    # Create meshgrid for pcolormesh
    extent = [x[0], x[-1], y[0], y[-1]]

    if args.log_scale:
        # Use symmetric log scale for data that may have negative values
        vmax = np.abs(data).max()
        norm = matplotlib.colors.SymLogNorm(linthresh=vmax / 100, vmin=-vmax, vmax=vmax)
        im = ax.imshow(data.T, origin='lower', aspect='auto', extent=extent, cmap=args.cmap, norm=norm)
    else:
        im = ax.imshow(data.T, origin='lower', aspect='auto', extent=extent, cmap=args.cmap)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"${data_obj.label}$ [{data_obj.units}]")

    # Labels
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"${data_obj.label}$ at t = {data_obj.time}")

    ax.set_xlabel(data_obj.axis[0].get('plot_label', f'x [{data_obj.axis[0]["units"]}]'))
    ax.set_ylabel(data_obj.axis[1].get('plot_label', f'y [{data_obj.axis[1]["units"]}]'))

    plt.tight_layout()

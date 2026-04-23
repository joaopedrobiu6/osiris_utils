from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_3d"]


def plot_3d(
    diagnostic,
    idx: int,
    scale_type: Literal["zero_centered", "pos", "neg", "default"] = "default",
    boundaries: np.ndarray = None,
):
    """
    Plots a 3D scatter plot of the diagnostic data (grid data).

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object containing the data.
    idx : int
        Index of the data to plot.
    scale_type : Literal["zero_centered", "pos", "neg", "default"], optional
        Type of scaling for the colormap:
        - "zero_centered": Center colormap around zero.
        - "pos": Colormap for positive values.
        - "neg": Colormap for negative values.
        - "default": Standard colormap.
    boundaries : np.ndarray, optional
        Boundaries to plot part of the data. (3,2) If None, uses the default grid boundaries.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object of the plot.
    """

    if diagnostic.dim != 3:
        raise ValueError("This method is only available for 3D diagnostics.")

    if boundaries is None:
        boundaries = diagnostic.grid

    if not isinstance(boundaries, np.ndarray):
        try:
            boundaries = np.array(boundaries)
        except Exception:
            boundaries = diagnostic.grid
            warnings.warn("boundaries cannot be accessed as a numpy array with shape (3, 2), using default instead", stacklevel=2)

    if boundaries.shape != (3, 2):
        warnings.warn("boundaries should have shape (3, 2), using default instead", stacklevel=2)
        boundaries = diagnostic.grid

    # Load data
    # Access private attributes if necessary, but prefer public properties if available
    # diagnostic.data is a property, but it might not be loaded for specific index if not load_all
    # The original code used self._all_loaded and self._data vs self[idx]

    # Check if we can access _all_loaded, otherwise assume we need to load
    if hasattr(diagnostic, "_all_loaded") and diagnostic._all_loaded:
        data = diagnostic._data[idx]
    else:
        data = diagnostic[idx]

    X, Y, Z = np.meshgrid(diagnostic.x[0], diagnostic.x[1], diagnostic.x[2], indexing="ij")

    # Flatten arrays for scatter plot
    (
        X_flat,
        Y_flat,
        Z_flat,
    ) = (
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
    )
    data_flat = data.ravel()

    # Apply filter: Keep only chosen points
    mask = (
        (X_flat > boundaries[0][0])
        & (X_flat < boundaries[0][1])
        & (Y_flat > boundaries[1][0])
        & (Y_flat < boundaries[1][1])
        & (Z_flat > boundaries[2][0])
        & (Z_flat < boundaries[2][1])
    )
    X_cut, Y_cut, Z_cut, data_cut = (
        X_flat[mask],
        Y_flat[mask],
        Z_flat[mask],
        data_flat[mask],
    )

    if scale_type == "zero_centered":
        # Center colormap around zero
        cmap = "seismic"
        vmax = np.max(np.abs(data_flat))  # Find max absolute value
        vmin = -vmax
    elif scale_type == "pos":
        cmap = "plasma"
        vmax = np.max(data_flat)
        vmin = 0

    elif scale_type == "neg":
        cmap = "plasma"
        vmax = 0
        vmin = np.min(data_flat)
    else:
        cmap = "plasma"
        vmax = np.max(data_flat)
        vmin = np.min(data_flat)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot with seismic colormap
    sc = ax.scatter(X_cut, Y_cut, Z_cut, c=data_cut, cmap=cmap, norm=norm, alpha=1)

    # Set limits to maintain full background
    ax.set_xlim(*diagnostic.grid[0])
    ax.set_ylim(*diagnostic.grid[1])
    ax.set_zlim(*diagnostic.grid[2])

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

    # Labels
    # Use public properties where possible
    cbar.set_label(rf"${diagnostic.name}$" + rf"$\  [{diagnostic.units}]$")

    # time(idx) returns [value, unit]
    t_val, t_unit = diagnostic.time(idx)
    ax.set_title(rf"$t={t_val:.2f}$" + rf"$\  [{t_unit}]$")

    ax.set_xlabel(r"${}$".format(diagnostic.axis[0]["long_name"]) + r"$\  [{}]$".format(diagnostic.axis[0]["units"]))
    ax.set_ylabel(r"${}$".format(diagnostic.axis[1]["long_name"]) + r"$\  [{}]$".format(diagnostic.axis[1]["units"]))
    ax.set_zlabel(r"${}$".format(diagnostic.axis[2]["long_name"]) + r"$\  [{}]$".format(diagnostic.axis[2]["units"]))

    return fig, ax

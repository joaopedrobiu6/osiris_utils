r"""Finite-difference derivatives: Derivative_Simulation and Derivative_Diagnostic.

A derivative is just another lazy ``Diagnostic``: ``Derivative_Simulation(sim,
"x1")["e1"][7]`` reads the frames it needs, differentiates, and returns the
array.  Nothing is precomputed and nothing is cached beyond a small ring of
input frames (``Derivative_Diagnostic.frame_cache_size``), which is what makes
a time derivative on a 3-D run possible at all.

Schemes available
-----------------
``order=2`` / ``order=4``
    Centred stencils in the interior, one-sided of the same order at the edges.
``stencil=[...]``
    Any set of integer offsets, e.g. ``[-2,-1,0,1,2]`` (centred) or
    ``[0,1,2,3]`` (forward).  Coefficients are solved from the Vandermonde
    system, so the scheme is exact for polynomials up to ``len(stencil)-1``.
``deriv_order=m``
    m-th derivative from that stencil (needs ``len(stencil) > m``).
``periodic=True``
    Wrap-around instead of one-sided edges — use it on the transverse axis.
``filter=SpatialFilter``
    Replace finite differences by the filter's own single-pass derivative
    kernel (Savitzky-Golay polynomial fit, derivative-of-Gaussian).  See
    example 06.

Run::

    python examples/scripts/04_derivatives.py [--sim DECK] [--plot]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou
from osiris_utils.postprocessing.derivative import _NUMBA_AVAILABLE


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]

    # ------------------------------------------------------------------
    section("1. The wrapper: one derivative applied to a whole simulation")
    # ------------------------------------------------------------------
    d_dx1 = ou.Derivative_Simulation(sim, "x1")  # default: order=4, non-periodic
    show("d/dx1 of e1 at t=3", d_dx1["e1"][3].shape)
    show("d/dx1 of a species moment", d_dx1[species]["vfl1"][3].shape)
    show("nothing was loaded", sim["e1"].all_loaded)

    # Results are cached per key, so asking twice does not rebuild the wrapper.
    show("cached", d_dx1["e1"] is d_dx1["e1"])
    d_dx1.delete("e1")
    d_dx1.delete_all()

    # ------------------------------------------------------------------
    section("2. Every deriv_type")
    # ------------------------------------------------------------------
    e1 = sim["e1"]
    for deriv_type, axis in [("t", None), ("x1", None), ("x2", None)]:
        if deriv_type == "x2" and e1.dim < 2:
            continue
        d = ou.Derivative_Diagnostic(e1, deriv_type, axis=axis)
        show(f"d/d{deriv_type}", d[3].shape)

    if e1.dim >= 2:
        # 'xx' takes a pair of spatial axes: d2/dx1 dx2 (axis=(1, 2))
        show("d2/dx1dx2 ('xx', axis=(1,2))", ou.Derivative_Diagnostic(e1, "xx", axis=(1, 2))[3].shape)
    # 'xt' = d/dt of d/dx_axis ; 'tx' = d/dx_axis of d/dt.  Same result for
    # smooth data, different intermediate — pick the one whose inner derivative
    # is the cheaper of the two on your grid.
    show("d2/dx1dt ('xt', axis=1)", ou.Derivative_Diagnostic(e1, "xt", axis=1)[3].shape)
    show("d2/dtdx1 ('tx', axis=1)", ou.Derivative_Diagnostic(e1, "tx", axis=1)[3].shape)

    # ------------------------------------------------------------------
    section("3. Accuracy — 2nd vs 4th order, against an exact derivative")
    # ------------------------------------------------------------------
    # The transverse direction of the synthetic run carries an exact cosine, so
    # d/dx2 has a closed form to compare against.  On your own run, compare the
    # two orders to each other instead.
    if e1.dim >= 2:
        b3 = sim["b3"]
        f = b3[3]
        dx2 = float(b3.dx[1])
        x2 = b3.x[1]

        # Spectral reference: exact for a band-limited periodic signal.
        k2 = 2 * np.pi * np.fft.fftfreq(len(x2), d=dx2)
        exact = np.fft.ifft(1j * k2[None, :] * np.fft.fft(f, axis=1), axis=1).real

        # With ~8 points per wavelength the errors should sit near
        # (k dx)^2 / 6 and (k dx)^4 / 30 — i.e. order 4 roughly 8x better.
        for order in (2, 4):
            d = ou.Derivative_Diagnostic(b3, "x2", order=order, periodic=True)
            err = np.abs(d[3] - exact).max() / np.abs(exact).max()
            show(f"periodic order={order} rel. error", f"{err:.3e}")

        # Non-periodic on a periodic axis: the one-sided edge stencils see a
        # boundary that is not there, so the error is concentrated at the edges.
        d_open = ou.Derivative_Diagnostic(b3, "x2", order=4, periodic=False)
        err_edge = np.abs(d_open[3] - exact)[:, :2].max() / np.abs(exact).max()
        err_bulk = np.abs(d_open[3] - exact)[:, 3:-3].max() / np.abs(exact).max()
        show("periodic=False: edge / bulk error", f"{err_edge:.3e} / {err_bulk:.3e}")

    # ------------------------------------------------------------------
    section("4. Custom stencils and higher derivative orders")
    # ------------------------------------------------------------------
    b3 = sim["b3"]
    show("centred 5-point, 1st derivative", ou.Derivative_Diagnostic(b3, "x1", stencil=[-2, -1, 0, 1, 2])[3].shape)
    show("forward 4-point (one-sided)", ou.Derivative_Diagnostic(b3, "x1", stencil=[0, 1, 2, 3])[3].shape)
    show("2nd derivative from 5 points", ou.Derivative_Diagnostic(b3, "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=2)[3].shape)
    show("4th derivative from 5 points", ou.Derivative_Diagnostic(b3, "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=4)[3].shape)

    # A stencil must have more points than the derivative order, and offsets
    # must be distinct — both are checked up front rather than producing noise.
    try:
        ou.Derivative_Diagnostic(b3, "x1", stencil=[0, 1], deriv_order=3)[3]
    except (ValueError, RuntimeError) as e:
        show("too few stencil points", type(e).__name__)

    # d2/dx2^2 the direct way vs. two chained first derivatives: same order of
    # accuracy, but one pass instead of two (and one edge treatment, not two).
    if e1.dim >= 2:
        direct = ou.Derivative_Diagnostic(b3, "x2", stencil=[-2, -1, 0, 1, 2], deriv_order=2, periodic=True)[3]
        once = ou.Derivative_Diagnostic(b3, "x2", order=4, periodic=True)
        chained = ou.Derivative_Diagnostic(once, "x2", order=4, periodic=True)[3]
        rel = np.abs(direct - chained).max() / np.abs(direct).max()
        show("direct d2 vs chained d1(d1), rel.", f"{rel:.3e}")

    # ------------------------------------------------------------------
    section("5. Chaining whole simulations")
    # ------------------------------------------------------------------
    # Derivative_Simulation accepts another Derivative_Simulation, so a mixed
    # second derivative is two wrappers deep and still lazy.
    d2 = ou.Derivative_Simulation(ou.Derivative_Simulation(sim, "x1"), "t")
    show("d/dt d/dx1 of e1", d2["e1"][3].shape)
    show("species handler through the chain", d2[species]["vfl1"][3].shape)

    # ------------------------------------------------------------------
    section("6. Time derivatives and burst dumps")
    # ------------------------------------------------------------------
    # The time stencil assumes frames are equally spaced (dt * ndump apart).
    # A run written with `if_use_burst_dump` is *not* equally spaced, and the
    # derivative refuses rather than returning a wrong number — build the
    # database with BurstConfig instead (example 11).
    d_dt = ou.Derivative_Diagnostic(sim[species]["vfl1"], "t", order=2)
    show("d/dt on a uniform series", d_dt[3].shape)
    show("frames kept per instance", ou.Derivative_Diagnostic.frame_cache_size)

    # ------------------------------------------------------------------
    section("7. load_all() on a derivative")
    # ------------------------------------------------------------------
    # Whole-array mode: reads every frame, then differentiates the stacked
    # array in one vectorised pass.  Spatial derivatives can be split over
    # processes with n_workers; numba, if installed, accelerates the stencil.
    show("numba available", _NUMBA_AVAILABLE)
    d = ou.Derivative_Diagnostic(sim["b3"], "x1", order=4)
    d.load_all(n_workers=2)
    show("d.data.shape", d.data.shape)
    show("same as the lazy path", np.allclose(d.data[3], ou.Derivative_Diagnostic(sim["b3"], "x1", order=4)[3]))
    d.unload()

    # ------------------------------------------------------------------
    section("8. A plot")
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    b3 = sim["b3"]
    d_b3 = ou.Derivative_Diagnostic(b3, "x1", order=4)
    line = b3[3] if b3.dim == 1 else b3[3].mean(axis=1)
    dline = d_b3[3] if b3.dim == 1 else d_b3[3].mean(axis=1)
    x = b3.x if b3.dim == 1 else b3.x[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 4.5))
    ax1.plot(x, line)
    ax1.set_ylabel(f"${b3.label}$")
    ax2.plot(x, dline, color="C1")
    ax2.set_ylabel(rf"$\partial_{{x_1}} {b3.label}$")
    ax2.set_xlabel(b3.axis[0]["plot_label"])
    fig.suptitle("field and its longitudinal derivative")
    savefig(fig, "04_derivative.png", args)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

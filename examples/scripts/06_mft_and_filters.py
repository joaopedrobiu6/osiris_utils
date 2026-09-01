r"""Mean-field decomposition and spatial filters.

Two related things, used together throughout the shock analysis:

**Mean field theory (MFT)** splits a quantity into its transverse average and
the fluctuation about it,

.. math:: f(x_1, x_2, t) = \langle f\rangle(x_1, t) + \delta f(x_1, x_2, t),

with :math:`\langle\cdot\rangle` the mean over ``mft_axis`` (OSIRIS 1-indexed;
2 = the transverse direction x2).  ``MFT_Diagnostic`` is a *container*: index it
with ``"avg"`` or ``"delta"`` to get the actual time-indexed diagnostics.

**Spatial filters** smooth a frame before any physics is computed, and — this
is the part that is easy to miss — each one also carries **its own derivative
scheme**:

* :class:`~osiris_utils.NoFilter` — identity smoothing, 4th-order finite
  differences.  The default everywhere; reproduces unfiltered results exactly.
* :class:`~osiris_utils.SavitzkyGolayFilter` — local polynomial fit;
  the m-th derivative is the m-th derivative *of that fit*, in one pass
  (needs ``polyorder >= m``).
* :class:`~osiris_utils.GaussianFilter` — convolution with the analytic
  m-th derivative-of-Gaussian kernel, again one pass.

One pass per derivative order is the point: chaining first derivatives would
apply the smoothing kernel m times and make the effective filter scale depend
on the derivative order.

Run::

    python examples/scripts/06_mft_and_filters.py [--sim DECK] [--plot]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou
from osiris_utils.filters import fd_derivative
from osiris_utils.postprocessing.filtering import Filtered_Diagnostic, Filtered_Simulation


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]
    b3 = sim["b3"]

    if b3.dim < 2:
        print("This example needs a 2-D run (it averages over the transverse axis).")
        return

    # ==================================================================
    section("1. MFT: average and fluctuation")
    # ==================================================================
    mft = ou.MFT_Simulation(sim, mft_axis=2)
    b3_mft = mft["b3"]  # a container, not a diagnostic
    avg = b3_mft["avg"]
    delta = b3_mft["delta"]

    show("<b3> shape (transverse axis collapsed)", avg[3].shape)
    show("delta b3 shape (full grid)", delta[3].shape)
    show("<b3> + delta b3 == b3", np.allclose(avg[3] + delta[3], b3[3], atol=1e-6))
    show("<delta b3> ~ 0", float(np.abs(delta[3].mean(axis=1)).max()))

    show("species quantities too", mft[species]["vfl1"]["avg"][3].shape)
    try:
        b3_mft["variance"]
    except ValueError as e:
        show("only 'avg' and 'delta'", f"ValueError: {e}")

    # Direct construction, and load_all() for the whole series at once.
    direct = ou.MFT_Diagnostic(sim[species]["n"], mft_axis=2)
    show("MFT_Diagnostic(...)['avg']", direct["avg"][3].shape)
    direct["avg"].load_all()
    show("<n>(x1, t)", direct["avg"].data.shape)

    mft.delete("b3")
    mft.delete_all()

    # ==================================================================
    section("2. The filters themselves (plain arrays)")
    # ==================================================================
    frame = np.asarray(b3[3], dtype=np.float64)
    dx1, dx2 = float(b3.dx[0]), float(b3.dx[1])

    filters = {
        "NoFilter": ou.NoFilter(),
        "SavitzkyGolay(9, 4)": ou.SavitzkyGolayFilter(window_length=9, polyorder=4),
        "Gaussian(sigma=1.5)": ou.GaussianFilter(sigma=1.5),
    }
    for name, filt in filters.items():
        # periodic is per axis: x1 open, x2 periodic — the shock convention.
        smoothed = filt.smooth(frame, periodic=(False, True))
        show(f"{name}: max|f - smooth(f)|", f"{np.abs(frame - smoothed).max():.4e}")

    # `axes=` restricts smoothing to chosen axes (0-indexed numpy axes).  A
    # normalised wrap-mode kernel applied along x2 only conserves the
    # transverse mean exactly, which is a clean way to see that x1 was left alone.
    x2_only = ou.GaussianFilter(sigma=1.5, axes=(1,))
    show(
        "Gaussian(axes=(1,)) preserves <f>_x2",
        np.allclose(x2_only.smooth(frame, periodic=(False, True)).mean(axis=1), frame.mean(axis=1)),
    )

    # truncate sets the kernel radius in units of sigma.  Derivative kernels of
    # order >= 2 internally widen it to at least 8 sigma: their low-k response
    # goes like k^m, so the tails a smoothing kernel can ignore dominate them.
    show("GaussianFilter repr", repr(ou.GaussianFilter(sigma=1.5, truncate=6.0)))

    # Calling a filter is smoothing: filt(f) == filt.smooth(f)
    show("__call__ == smooth", np.array_equal(filters["Gaussian(sigma=1.5)"](frame), filters["Gaussian(sigma=1.5)"].smooth(frame)))

    # ==================================================================
    section("3. Filter derivatives vs finite differences")
    # ==================================================================
    # Exact reference for the periodic transverse axis.
    k2 = 2 * np.pi * np.fft.fftfreq(frame.shape[1], d=dx2)
    exact = np.fft.ifft(1j * k2[None, :] * np.fft.fft(frame, axis=1), axis=1).real

    # These are NOT all approximations of the same thing: a smoothing filter's
    # derivative is the derivative of the *smoothed* field.  On this grid the
    # transverse mode is resolved by only ~8 cells, so a sigma = 1.5-cell
    # Gaussian removes a large part of it — the "error" below is the filter
    # doing its job, and is the price you pay for suppressing PIC noise.
    for name, filt in filters.items():
        d = filt.derivative(frame, dx2, axis=1, order=1, periodic=True)
        err = np.abs(d - exact).max() / np.abs(exact).max()
        show(f"{name} d/dx2 vs exact d/dx2", f"{err:.3e}")

    # fd_derivative is the standalone 4th-order stencil NoFilter uses; higher
    # orders are repeated applications of the first-derivative scheme.
    show("fd_derivative order=2", fd_derivative(frame, dx1, axis=0, order=2, periodic=False).shape)

    # Savitzky-Golay refuses a derivative its polynomial fit cannot support.
    try:
        ou.SavitzkyGolayFilter(9, 2).derivative(frame, dx1, axis=0, order=3)
    except ValueError as e:
        show("SavGol polyorder < order", f"ValueError: {str(e)[:60]}...")

    # ==================================================================
    section("4. FilterChain and as_filter")
    # ==================================================================
    chain = ou.FilterChain(ou.GaussianFilter(sigma=1.0), ou.SavitzkyGolayFilter(9, 4))
    show("chain", repr(chain))
    show("chain.filters", len(chain.filters))
    # smooth() runs every filter in order; derivative() uses the LAST filter's
    # kernel only — in the pipeline it is always called on already-smoothed
    # data, so re-applying the earlier filters would smooth twice.
    show("chain.smooth", chain.smooth(frame, periodic=(False, True)).shape)
    show("chain.derivative (last filter's kernel)", chain.derivative(frame, dx1, axis=0, order=1).shape)

    # as_filter normalises whatever a config carries into one SpatialFilter.
    show("as_filter(None)", repr(ou.as_filter(None)))
    show("as_filter(())", repr(ou.as_filter(())))
    show("as_filter([f])", repr(ou.as_filter([ou.GaussianFilter(1.0)])))
    show("as_filter([f, g])", type(ou.as_filter([ou.GaussianFilter(1.0), ou.NoFilter()])).__name__)

    # ==================================================================
    section("5. Filtering inside the lazy pipeline")
    # ==================================================================
    # Filtered_Simulation smooths every *raw* quantity on its way out of disk.
    # Nothing is precomputed: the cost is paid per frame you ask for.
    filt = ou.SavitzkyGolayFilter(window_length=9, polyorder=4)
    smooth_sim = Filtered_Simulation(sim, filt, mft_axis=2)

    show("smoothed field frame", smooth_sim["b3"][3].shape)
    show("smoothed moment frame", smooth_sim[species]["vfl1"][3].shape)
    show("equals filt.smooth(raw)", np.allclose(smooth_sim["b3"][3], filt.smooth(np.asarray(b3[3], np.float64), periodic=(False, True))))
    show("filter / periodic axes", (repr(smooth_sim.filter), smooth_sim.periodic_axes))

    # periodic_axes overrides the default derived from mft_axis.
    open_sim = Filtered_Simulation(sim, filt, periodic_axes=())
    show("all-open boundaries", open_sim["b3"][3].shape)

    # One diagnostic at a time, with explicit boundaries:
    show("Filtered_Diagnostic", Filtered_Diagnostic(b3, filt, periodic_axes=(2,))[3].shape)

    # A filtered view only smooths *raw* quantities.  Anything derived is
    # already built from smoothed leaves, so asking the view for a derived
    # diagnostic that lives on the wrapped simulation is an error rather than
    # a silent second smoothing pass.
    sim.add_diagnostic(sim[species]["n"] * sim[species]["T11"], "nT11_raw")
    try:
        smooth_sim["nT11_raw"]
    except KeyError as e:
        show("derived quantity on a filtered view", f"KeyError: {str(e)[:70]}...")
    # Register it on the view instead — it is handed back untouched.
    smooth_sim.add_diagnostic(smooth_sim[species]["n"] * smooth_sim[species]["T11"], "nT11")
    show("registered on the view", smooth_sim["nT11"][3].shape)

    # ==================================================================
    section("6. The other half: filtered derivatives")
    # ==================================================================
    # The database pipeline does two things per frame — smooth the field, then
    # differentiate the smoothed field with the filter's own kernel.  In the
    # lazy pipeline that is Filtered_* for the first half and
    # Derivative_Diagnostic(filter=...) for the second.
    d_filtered = ou.Derivative_Diagnostic(smooth_sim["b3"], "x1", deriv_order=1, filter=filt)
    reference = filt.derivative(filt.smooth(np.asarray(b3[3], np.float64), periodic=(False, True)), dx1, axis=0, order=1)
    show("smoothed then differentiated", np.allclose(d_filtered[3], reference))

    # A spatial kernel means nothing along the time axis, so this is refused.
    try:
        ou.Derivative_Diagnostic(sim[species]["vfl1"], "t", filter=filt)
    except ValueError as e:
        show("filter + deriv_type='t'", f"ValueError: {str(e)[:60]}...")

    # ==================================================================
    section("7. A plot")
    # ==================================================================
    import matplotlib.pyplot as plt

    # Use a species moment, not a field: the moments carry particle noise (in a
    # real run, from finite macro-particle statistics), which is exactly what
    # the filters are for.  Fields are already smooth.
    vfl1 = sim[species]["vfl1"]
    raw_frame = np.asarray(vfl1[3], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    # The raw transverse mean is exactly MFT's "avg" — averaging over 16 cells
    # already halves the noise; smoothing along x1 removes what is left.
    ax.plot(vfl1.x[0], ou.MFT_Simulation(sim, 2)[species]["vfl1"]["avg"][3].ravel(), "k", lw=0.8, label=r"raw $\langle v_1\rangle$")
    for name, f in [("SavGol(9,4)", filt), ("Gaussian(2.0)", ou.GaussianFilter(2.0))]:
        ax.plot(vfl1.x[0], f.smooth(raw_frame, periodic=(False, True)).mean(axis=1), lw=1.6, label=name)
    ax.set_xlabel(vfl1.axis[0]["plot_label"])
    ax.set_ylabel(r"$v_1$")
    ax.set_title("transverse mean of a noisy moment, raw and smoothed")
    ax.legend()
    savefig(fig, "06_filters.png", args)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

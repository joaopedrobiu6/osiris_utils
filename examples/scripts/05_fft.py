r"""FFTs of a diagnostic: FFT_Simulation and FFT_Diagnostic.

``FFT_Diagnostic`` returns the **power spectrum** :math:`|\hat f|^2`, shifted so
the zero frequency sits in the middle of the array (``np.fft.fftshift``).

Two modes, and the difference matters:

* **spatial only** (``fft_axis=1`` or ``[1, 2]``) — computed per frame, lazily.
  ``fft[7]`` transforms just that dump.
* **including time** (``0 in fft_axis``) — needs the whole series, so you must
  call ``load_all()``; a single frame cannot know about time.

Preprocessing options, all of them physical choices rather than cosmetics:

``detrend="mean"``
    Subtract the mean over the transform axes first.  Without it a non-zero
    background dominates the k=0 bin and its leakage buries everything else.
``window="hann"`` / ``window_for_spatial``
    Taper before transforming (``"hann"``/``"hanning"``, or ``None`` for no
    taper — those are the only values implemented).  Required on a **non**-
    periodic axis (a jump between the two ends leaks across all k); wrong on a
    periodic one, where it destroys an otherwise exact transform.
``assume_periodic=True``
    Sets the spatial default to *no* window — the right choice when the
    transverse direction really is periodic.
``normalize``
    ``"ortho"`` (unitary, default), ``"none"`` (numpy's raw scaling), or
    ``"density"`` (multiplies by the grid spacings, giving a spectral density).

Run::

    python examples/scripts/05_fft.py [--sim DECK] [--plot]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]
    b3 = sim["b3"]

    # ------------------------------------------------------------------
    section("1. Spatial FFT, one frame at a time")
    # ------------------------------------------------------------------
    fft_x1 = ou.FFT_Diagnostic(b3, fft_axis=1)  # x1 only
    show("fft over x1, frame 3", fft_x1[3].shape)
    show("input was not loaded", b3.all_loaded)

    if b3.dim >= 2:
        fft_x2 = ou.FFT_Diagnostic(b3, fft_axis=2, assume_periodic=True)
        show("fft over x2 (periodic, no window)", fft_x2[3].shape)
        fft_xy = ou.FFT_Diagnostic(b3, fft_axis=[1, 2])
        show("2-D fft over (x1, x2)", fft_xy[3].shape)

    # Spatial slicing works, but a cropped window is no longer periodic, so a
    # Hann taper is forced on when you slice and asked for no window.
    show("sliced fft", ou.FFT_Diagnostic(b3, fft_axis=1)[3, 8:40].shape)

    # ------------------------------------------------------------------
    section("2. The axes: k and omega")
    # ------------------------------------------------------------------
    # k() and omega() read the transformed array's shape, so they need the data
    # in memory: call load_all() before asking for an axis.
    fft_x1.load_all()
    k1 = fft_x1.k(1)
    show("k(1)", k1)
    show("k() -> dict per axis", {ax: v.shape for ax, v in fft_x1.k().items()})
    show("kmax (Nyquist)", fft_x1.kmax)
    show("omega_max (Nyquist)", fft_x1.omega_max)

    # The synthetic run carries an exact transverse mode; the spectrum finds it.
    if b3.dim >= 2:
        fft_x2 = ou.FFT_Diagnostic(b3, fft_axis=2, assume_periodic=True, detrend="mean")
        fft_x2.load_all()
        k2 = fft_x2.k(2)
        power = fft_x2.data[3].mean(axis=0)
        peak = k2[np.argmax(power)]
        show("dominant k2 found", f"{abs(peak):.4f}")
        show("expected 2*pi*m/L2", f"{2 * np.pi * 2 / (b3.grid[1][1] - b3.grid[1][0]):.4f}")

    # ------------------------------------------------------------------
    section("3. FFT including the time axis")
    # ------------------------------------------------------------------
    # fft_axis=[0, 1] gives an (omega, k) dispersion map; 0 alone is a pure
    # temporal spectrum per grid point.
    fft_wk = ou.FFT_Diagnostic(b3, fft_axis=[0, 1], window="hann", detrend="mean")
    fft_wk.load_all()
    show("|F(omega, k1, x2)|^2", fft_wk.data.shape)
    show("omega axis", fft_wk.omega())
    show("k1 axis", fft_wk.k(1).shape)

    # A single frame cannot carry a time transform, so indexing one that has
    # not been loaded raises.  (After load_all() indexing just slices the
    # array that is already in memory, which is why this uses a fresh object.)
    try:
        ou.FFT_Diagnostic(b3, fft_axis=[0, 1])[3]
    except (ValueError, RuntimeError) as e:
        show("indexing an unloaded time FFT", type(e).__name__)

    # ------------------------------------------------------------------
    section("4. Preprocessing options side by side")
    # ------------------------------------------------------------------
    frame_opts = [
        ("default (hann, detrend=mean, ortho)", {}),
        ("no window", {"window": None}),
        ("no detrend", {"detrend": None}),
        ("normalize='none'", {"normalize": "none"}),
        ("normalize='density'", {"normalize": "density"}),
        ("assume_periodic=True", {"assume_periodic": True}),
        # Only "hann"/"hanning" (or None/"none") are implemented; anything
        # else raises.  window_for_spatial overrides the default that
        # assume_periodic would pick for the spatial axes.
        ("explicit spatial window", {"assume_periodic": True, "window_for_spatial": "hann"}),
    ]
    for label, kwargs in frame_opts:
        spec = ou.FFT_Diagnostic(b3, fft_axis=1, **kwargs)[3]
        show(label, f"total power {spec.sum():.4e}")

    # ------------------------------------------------------------------
    section("5. The Simulation wrapper")
    # ------------------------------------------------------------------
    fft_sim = ou.FFT_Simulation(sim, fft_axis=1)
    show("fft_sim['e1'][3]", fft_sim["e1"][3].shape)
    show("fft_sim[species]['vfl1'][3]", fft_sim[species]["vfl1"][3].shape)
    # `process` applies the wrapper's settings to a Diagnostic you built
    # yourself — e.g. a product of two quantities.
    show("process(n * vfl1)", fft_sim.process(sim[species]["n"] * sim[species]["vfl1"])[3].shape)
    fft_sim.delete("e1")
    fft_sim.delete_all()

    # ------------------------------------------------------------------
    section("6. A plot")
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    spectrum = fft_x1.data[3]
    if spectrum.ndim > 1:
        spectrum = spectrum.mean(axis=1)
    positive = k1 >= 0

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.semilogy(k1[positive], spectrum[positive])
    ax.set_xlabel(r"$k_1\ [\omega_p/c]$")
    ax.set_ylabel(rf"$|\widehat{{{b3.label}}}(k_1)|^2$")
    ax.set_title("longitudinal power spectrum")
    savefig(fig, "05_spectrum.png", args)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

r"""The rest: planning helpers, array utilities, profiling and 3-D plotting.

Small things that do not belong to the Diagnostic pipeline:

* **Run planning** — Courant limit, wall-time and file-size estimates, for
  sizing a job before submitting it.
* **Array utilities** — transverse average, cumulative integration, save/read.
* **Profiling** — opt-in, and genuinely zero-overhead when off: every timing
  call sits behind an ``isEnabledFor(DEBUG)`` check on the
  ``osiris_utils.profile`` logger.
* **3-D plotting** — ``ou.vis.plot_3d`` scatters a 3-D grid diagnostic.

Run::

    python examples/scripts/12_utils_profiling_vis.py [--sim DECK] [--plot]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou
from osiris_utils.utils import resolve_rqm
from osiris_utils.vis import plot_3d


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    section("1. Planning a run")
    # ==================================================================
    # Courant limit for a 2-D Yee solver: dt < 1 / sqrt(1/dx^2 + 1/dy^2).
    dx, dy = 0.05, 0.05
    show("courant2D(0.05, 0.05)", ou.courant2D(dx, dy))
    show("...for dy = 0.10", ou.courant2D(0.05, 0.10))

    # Wall time from the push count: cells * ppc * steps * push_time / cpus.
    # push_time is the per-particle push cost of your machine (~1e-7 s is a
    # reasonable starting guess; measure it once and keep the number).
    show(
        "time_estimation(2048^2 cells, 64 ppc, 10k steps, 1024 cpu)",
        f"{ou.time_estimation(2048**2, 64, 10_000, 1024, hours=True):.2f} h",
    )
    show("...with a faster push (5e-8 s)", f"{ou.time_estimation(2048**2, 64, 10_000, 1024, push_time=5e-8, hours=True):.2f} h")

    # Size of one single-precision grid dump, in MB.
    show("filesize_estimation(2048^2)", f"{ou.filesize_estimation(2048**2):.1f} MB")
    show("...times 500 dumps", f"{500 * ou.filesize_estimation(2048**2) / 1024:.1f} GB")

    # ==================================================================
    section("2. Array utilities")
    # ==================================================================
    e1 = sim["e1"]
    frame = e1[3]

    if frame.ndim == 2:
        # The mean over the second axis of a 2-D frame — the same operation
        # MFT's "avg" performs, for when you have a bare array.
        show("transverse_average", ou.transverse_average(frame).shape)
        try:
            ou.transverse_average(frame[:, 0])
        except ValueError as e:
            show("...only takes 2-D", f"ValueError: {e}")

    # Cumulative integration (Simpson where SciPy has it, trapezoid otherwise).
    # start_side picks which end is zero: 'right' integrates from the right
    # boundary inwards (the usual choice for a potential that vanishes
    # downstream), 'left' from the left.
    profile = frame.mean(axis=1) if frame.ndim == 2 else frame
    dx1 = float(e1.dx[0]) if np.ndim(e1.dx) else float(e1.dx)
    from_right = ou.integrate(profile, dx1)
    from_left = ou.integrate(profile, dx1, start_side="left")
    show("integrate (from the right)", (from_right[0], from_right[-1]))
    show("integrate (from the left)", (from_left[0], from_left[-1]))
    # The two differ by the total integral (a constant), up to the difference
    # between integrating the array and integrating it reversed.
    show("spread of (right - left)", f"{float(np.ptp(from_right - from_left)):.3e}")

    # N-D arrays need the axis:
    if frame.ndim == 2:
        show("integrate along axis=0", ou.integrate(frame, dx1, axis=0).shape)

    # Text round-trips (numpy .txt or pandas .csv).
    ou.save_data(profile, str(args.outdir / "profile.txt"))
    show("save_data / read_data", np.allclose(ou.read_data(str(args.outdir / "profile.txt")), profile))

    # m/q of a species, straight from the deck — the number that decides the
    # sign and mass scaling of every inertial term in the momentum equation.
    show(f"resolve_rqm({species})", resolve_rqm(sim, species))
    show("with an override", resolve_rqm(sim, species, override=-1.0))

    # ==================================================================
    section("3. Profiling")
    # ==================================================================
    # Off by default and free: the timing calls are behind a level check.
    logfile = args.outdir / "profile.log"
    ou.enable_profiling(logfile=str(logfile))

    with ou.profile_block("read 4 frames of e1"):
        for i in range(4):
            _ = e1[i]

    with ou.profile_block("load_all e1"):
        e1.load_all(use_parallel=False)
    e1.unload()

    ou.disable_profiling()
    # The records also reach the root handler the package installs, which is
    # why they appear on stderr as well as in the file.
    show("profile log", logfile.read_text().strip().splitlines()[-2:])
    # Every load_all() and per-frame read is instrumented internally too, so
    # enabling the logger alone already gives a breakdown.  On HPC, give each
    # rank its own file: enable_profiling(logfile=f"profile_rank{rank}.log").

    # ==================================================================
    section("4. 3-D visualisation")
    # ==================================================================
    # plot_3d scatters the cells of a 3-D grid diagnostic, coloured by value.
    # scale_type picks the colour normalisation:
    #   "zero_centered" — symmetric about 0 (fields that change sign)
    #   "pos" / "neg"   — one-signed data
    #   "default"       — min..max
    if args.sim:
        show("3-D demo skipped", "pass --sim with a 3-D run to try it")
    else:
        sim3d = load_simulation(args, ndims=3)
        e3 = sim3d["e3"]
        show("3-D diagnostic", (e3.dim, e3[1].shape))

        # plot_3d lives in osiris_utils.vis and is not re-exported at the top
        # level: `from osiris_utils.vis import plot_3d`.
        fig, ax = plot_3d(e3, idx=1, scale_type="zero_centered")
        savefig(fig, "12_plot3d.png", args)

        # `boundaries` (3, 2) plots only part of the box — essential when the
        # full grid is too many points to scatter.
        half = e3.grid.copy()
        half[0, 1] = 0.5 * (half[0, 0] + half[0, 1])
        fig2, _ = plot_3d(e3, idx=1, scale_type="default", boundaries=half)
        savefig(fig2, "12_plot3d_half.png", args)

        import matplotlib.pyplot as plt

        plt.close("all")

    # ==================================================================
    section("5. Tuning the frame cache")
    # ==================================================================
    # Each Diagnostic keeps its last few frames, so an expression that mentions
    # the same quantity many times (the anomalous-resistivity terms mention the
    # density a dozen times) reads each file once instead of once per mention.
    show("default frames kept", ou.Diagnostic.frame_cache_size)
    show("derivatives keep more (time stencils)", ou.Derivative_Diagnostic.frame_cache_size)
    b3 = sim["b3"]
    b3.frame_cache_size = 8  # per instance; assign on the class for all of them
    show("per-instance override", b3.frame_cache_size)
    # Set it to 0 when memory is tighter than I/O.

    print("\nDone.")


if __name__ == "__main__":
    main()

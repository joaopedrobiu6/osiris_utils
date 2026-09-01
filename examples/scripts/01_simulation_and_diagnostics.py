"""Simulation, Species_Handler and Diagnostic — the core of osiris_utils.

Everything else in the package is built on the objects shown here:

* :class:`osiris_utils.Simulation` — a run, addressed by its input deck.
* :class:`osiris_utils.Species_Handler` — ``sim["electrons"]``, the per-species
  quantities.
* :class:`osiris_utils.Diagnostic` — one quantity over time.  **Nothing is read
  from disk until a frame is asked for**, and arithmetic on Diagnostics stays
  lazy, so ``(e1 * vfl1)[7]`` reads exactly the two frames it needs.

Run::

    python examples/scripts/01_simulation_and_diagnostics.py [--sim DECK] [--plot]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou
from osiris_utils.data.diagnostic import OSIRIS_ALL, which_quantities


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)

    # ------------------------------------------------------------------
    section("1. Opening a run")
    # ------------------------------------------------------------------
    # Simulation takes the *input deck*; the run folder is the deck's parent.
    show("species in the deck", sim.species)
    show("loaded (in-memory) diagnostics", sim.loaded_diagnostics)

    # Every quantity name osiris_utils understands: fields (e1..b3, part_*,
    # ext_*), species reports (charge, j1..j3, ene), UDIST moments (vfl, ufl,
    # P, T, Q) and phase spaces (p1x1, gamma, ...).  `which_quantities` prints
    # them; `OSIRIS_ALL` is the list itself.  Neither is re-exported at the top
    # level -- import them from osiris_utils.data.diagnostic.
    which_quantities()
    show("number of known quantities", len(OSIRIS_ALL))

    # ------------------------------------------------------------------
    section("2. Getting diagnostics")
    # ------------------------------------------------------------------
    e1 = sim["e1"]  # a field: not species-dependent
    n = sim["electrons"]["n"]  # density: via the species handler
    vfl1 = sim["electrons"]["vfl1"]  # a UDIST moment
    show("sim['e1']", e1)
    show("sim['electrons']['n']", n)

    # Asking twice returns the *same* object — building one globs the dump
    # directory and opens a file for metadata, so it is cached.
    show("sim['e1'] is sim['e1']", sim["e1"] is e1)

    # ------------------------------------------------------------------
    section("3. Metadata (read from the first dump, or the deck)")
    # ------------------------------------------------------------------
    show("name / quantity", f"{e1.name} / {e1.quantity}")
    show("label (LaTeX)", e1.label)
    show("units (LaTeX)", e1.units)
    show("time units", e1.tunits)
    show("dim", e1.dim)
    show("nx (grid points)", e1.nx)
    show("dx (grid spacing)", e1.dx)
    show("grid (min, max per axis)", e1.grid)
    show("x (coordinates)", e1.x[0] if e1.dim > 1 else e1.x)
    show("dt", e1.dt)
    show("ndump", e1.ndump)
    show("maxiter == len(diag)", (e1.maxiter, len(e1)))
    show("type", e1.type)
    show("path", e1.path)
    show("simulation_folder", e1.simulation_folder)
    show("all_loaded", e1.all_loaded)
    show("axis[0]", e1.axis[0])

    # The true OSIRIS iteration of every frame.  This is *not* index * ndump
    # when the report used burst dumps, so always go through `iterations`.
    show("iterations", e1.iterations)
    show("index_of_iteration(iterations[3])", e1.index_of_iteration(int(e1.iterations[3])))
    show("time(3) -> [t, units]", e1.time(3))
    show("file_list[:2]", [p.split("/")[-1] for p in (e1.file_list or [])[:2]])

    # ------------------------------------------------------------------
    section("4. Reading frames — the lazy path")
    # ------------------------------------------------------------------
    show("e1[3].shape (one dump)", e1[3].shape)
    show("e1[-1] (negative index)", e1[-1].shape)
    show("e1[0:4].shape (stacked)", e1[0:4].shape)
    show("e1[::2].shape (strided)", e1[::2].shape)

    # Spatial slicing goes *into the HDF5 read*: only the requested block is
    # pulled off disk, which is what makes a 3-D run tractable.
    if e1.dim >= 2:
        show("e1[3, :, 4:8].shape", e1[3, :, 4:8].shape)
        show("e1[0:4, 10:20].shape", e1[0:4, 10:20].shape)

    # ------------------------------------------------------------------
    section("5. Reading everything — load_all()")
    # ------------------------------------------------------------------
    # Only when the whole series fits in memory.  A 3-D run does not.
    show("estimated size (MB)", e1[0].nbytes * len(e1) / 1024**2)

    e1.load_all()  # auto: parallel above 10 frames of >1 MB, sequential below
    show("after load_all: data.shape", e1.data.shape)
    show("all_loaded", e1.all_loaded)
    show("now registered on the Simulation", list(sim.loaded_diagnostics))

    e1.unload()  # frees the array *and* the per-frame LRU cache
    show("after unload: all_loaded", e1.all_loaded)

    # The explicit forms:
    #   sequential (small runs, or when the filesystem is the bottleneck)
    e1.load_all(use_parallel=False)
    e1.unload()
    #   threads — the right choice for HDF5: h5py releases the GIL, and the
    #   worker count is capped so a Lustre/GPFS metadata server is not flooded
    e1.load_all(use_parallel=True, executor_type="thread", n_workers=4)
    e1.unload()
    #   processes — only for a *base* Diagnostic, and only when each frame
    #   carries real CPU work.  A post-processed Diagnostic refuses, because
    #   the workers would re-read the file and skip the transformation.
    e1.load_all(use_parallel=True, executor_type="process", n_workers=2)
    show("process pool result", e1.data.shape)
    e1.unload()

    # ------------------------------------------------------------------
    section("6. Arithmetic — new Diagnostics, still lazy")
    # ------------------------------------------------------------------
    flux = n * vfl1  # Diagnostic * Diagnostic
    show("(n * vfl1)[3].shape", flux[3].shape)
    # Metadata comes from the left operand — a product has no derivable name,
    # label or units, so set them yourself (section 7).
    show("inherited name", flux.name)
    show("nothing was loaded", (n.all_loaded, vfl1.all_loaded))

    show("scalar ops", (2 * e1 + 1.0)[3].shape)
    show("division", (n / vfl1)[3].shape)
    show("power", (n**2)[3].shape)
    show("negation / abs", ((-n)[3].max(), abs(n)[3].min()))
    show("right-hand ops", (1.0 - n)[3].shape)
    show("with a numpy array", (n * np.ones_like(n[0]))[3].shape)

    # Chains stay lazy: this reads 4 frames, not 4 x maxiter.
    energy_flux = 0.5 * n * vfl1**2
    show("0.5 * n * vfl1**2 at t=3", float(energy_flux[3].mean()))

    # ------------------------------------------------------------------
    section("7. Custom diagnostics on the Simulation")
    # ------------------------------------------------------------------
    sim.add_diagnostic(flux, "nvfl1")
    show("sim['nvfl1'][3].shape", sim["nvfl1"][3].shape)
    sim["electrons"].add_diagnostic(n * sim["electrons"]["T11"], "nT11")
    show("sim['electrons']['nT11']", sim["electrons"]["nT11"][3].shape)

    # Metadata is inherited from the operands but is yours to correct — the
    # label and units of a product are not deducible.
    flux.name = "nvfl1"
    flux.label = r"n v_1"
    flux.units = r"c \omega_p^2 / c"
    show("relabelled", f"{flux.name}: ${flux.label}$ [{flux.units}]")

    sim.delete_diagnostic("nvfl1")
    show("after delete_diagnostic", list(sim.loaded_diagnostics))
    sim.delete_all_diagnostics()

    # ------------------------------------------------------------------
    section("8. Writing a diagnostic back out as OSIRIS HDF5")
    # ------------------------------------------------------------------
    # Files land in <run>/MS/MISC/<postprocess or DIR_name>/<savename>/ unless
    # `path` is given, and are readable again by OsirisGridFile / Simulation.
    out = args.outdir / "h5"
    flux.to_h5(savename="nvfl1", index=[0, 1, 2], path=str(out), verbose=False)
    written = sorted(p.name for p in out.glob("*.h5"))
    show("to_h5 wrote", written)
    show("read back", ou.OsirisGridFile(str(out / written[0])).data.shape)
    # flux.to_h5(savename="nvfl1", all=True)   # every timestep, default folder

    # ------------------------------------------------------------------
    section("9. A plot")
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    frame = e1[3]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if e1.dim == 1:
        ax.plot(e1.x, frame)
        ax.set_ylabel(f"${e1.label}$  $[{e1.units}]$")
    else:
        im = ax.pcolormesh(e1.x[0], e1.x[1], frame.T, shading="auto", cmap="RdBu_r")
        fig.colorbar(im, ax=ax, label=f"${e1.label}$  $[{e1.units}]$")
        ax.set_ylabel(e1.axis[1]["plot_label"])
    ax.set_xlabel(e1.axis[0]["plot_label"])
    t, tunits = e1.time(3)
    ax.set_title(rf"${e1.label}$ at $t = {t:g}\ [{tunits}]$")
    savefig(fig, "01_field.png", args)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

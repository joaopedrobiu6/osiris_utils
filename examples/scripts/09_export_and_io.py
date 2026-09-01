r"""Getting data out: parallel .npy export, HDF5 round-trips, plain text.

``export_to_npy`` is the workhorse for handing a run to something else (a
notebook, a training script, a collaborator).  Three properties matter:

* **It scales.**  Output is a ``np.lib.format.open_memmap`` written straight to
  disk, so peak RAM is ``chunk_size x frame_size`` — 5 000 timesteps costs the
  same as 50.
* **It is parallel.**  Frames are read by a thread pool; h5py releases the GIL,
  so the reads genuinely overlap.  Workers are capped at 32 to keep a Lustre or
  GPFS metadata server from being flooded.
* **It reduces on the fly.**  ``reduce_axis`` averages *within* each frame in
  the reader thread (only the small array crosses the thread boundary), and
  ``time_average`` streams a running mean with a float64 accumulator.

============================  ====================  ==================
``reduce_axis``               ``time_average``      output shape
============================  ====================  ==================
``None``                      ``False``             ``(n_t, nx1, nx2)``
``0``                         ``False``             ``(n_t, nx2)``
``1``                         ``False``             ``(n_t, nx1)``
``(0, 1)``                    ``False``             ``(n_t,)``
``None``                      ``True``              ``(nx1, nx2)``
``0``                         ``True``              ``(nx2,)``
============================  ====================  ==================

Run::

    python examples/scripts/09_export_and_io.py [--sim DECK]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, section, show

import osiris_utils as ou


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]
    out = args.outdir / "export"
    out.mkdir(parents=True, exist_ok=True)

    e1 = sim["e1"]

    # ==================================================================
    section("1. A full export")
    # ==================================================================
    path = ou.export_to_npy(e1, out / "e1.npy", overwrite=True, show_progress=False)
    show("written", path.name)
    # mmap_mode keeps it out of RAM on the way back in, too.
    show("shape (n_t, nx1, nx2)", np.load(path, mmap_mode="r").shape)
    show("matches the lazy read", np.allclose(np.load(path)[3], e1[3]))

    # ==================================================================
    section("2. Reductions")
    # ==================================================================
    p = ou.export_to_npy(e1, out / "e1_x1avg.npy", reduce_axis=0, overwrite=True, show_progress=False)
    show("reduce_axis=0  (average over x1)", np.load(p).shape)

    if e1.dim >= 2:
        p = ou.export_to_npy(e1, out / "e1_x2avg.npy", reduce_axis=1, overwrite=True, show_progress=False)
        show("reduce_axis=1  (average over x2)", np.load(p).shape)
        show("== the transverse mean", np.allclose(np.load(p)[3], e1[3].mean(axis=1), atol=1e-6))

        p = ou.export_to_npy(e1, out / "e1_scalar.npy", reduce_axis=(0, 1), overwrite=True, show_progress=False)
        show("reduce_axis=(0,1) (box average)", np.load(p).shape)

    p = ou.export_to_npy(e1, out / "e1_tavg.npy", time_average=True, overwrite=True, show_progress=False)
    show("time_average=True", np.load(p).shape)
    show("== the mean over dumps", np.allclose(np.load(p), e1[:].mean(axis=0), atol=1e-6))

    p = ou.export_to_npy(e1, out / "e1_both.npy", reduce_axis=0, time_average=True, overwrite=True, show_progress=False)
    show("both reductions", np.load(p).shape)

    # ==================================================================
    section("3. Tuning the export")
    # ==================================================================
    # n_workers   — reader threads (default min(32, n_t, cpu_count))
    # chunk_size  — frames in flight; peak RAM = chunk_size x frame bytes
    # checkpoint  — flush + a .progress sidecar after every chunk, so an
    #               interrupted job resumes.  Turn it off for one syscall less
    #               per chunk on a high-latency filesystem.
    # overwrite   — refuse to clobber an existing file unless True
    p = ou.export_to_npy(
        e1,
        out / "e1_tuned.npy",
        n_workers=4,
        chunk_size=16,
        checkpoint=False,
        show_progress=False,
        overwrite=True,
    )
    show("tuned export", np.load(p).shape)

    try:
        ou.export_to_npy(e1, out / "e1_tuned.npy", show_progress=False)
    except FileExistsError as e:
        show("overwrite=False refuses", f"FileExistsError: {str(e)[:52]}...")

    # ==================================================================
    section("4. Post-processed diagnostics export too")
    # ==================================================================
    # Anything with __len__ and __getitem__ works, so a derivative, a filtered
    # view or an arithmetic expression can be exported directly.
    d = ou.Derivative_Diagnostic(sim["b3"], "x1", order=4)
    p = ou.export_to_npy(d, out / "db3_dx1.npy", overwrite=True, show_progress=False)
    show("derivative exported", np.load(p).shape)

    flux = sim[species]["n"] * sim[species]["vfl1"]
    p = ou.export_to_npy(flux, out / "nvfl1.npy", reduce_axis=1 if flux.dim > 1 else None, overwrite=True, show_progress=False)
    show("expression exported", np.load(p).shape)

    # ==================================================================
    section("5. Several quantities at once")
    # ==================================================================
    # Species quantities use "species/quantity" and are saved as
    # <species>_<quantity>.npy.  Quantities go one at a time so the thread pool
    # is fully used per quantity instead of competing for bandwidth.
    written = ou.export_simulation_to_npy(
        sim,
        quantities=["e1", "b3", f"{species}/n", f"{species}/vfl1"],
        output_folder=out / "bulk",
        reduce_axis=1 if e1.dim > 1 else None,
        overwrite=True,
        show_progress=False,
    )
    for key, value in written.items():
        show(key, f"{value.name}  {np.load(value).shape}")

    # ==================================================================
    section("6. Back to OSIRIS HDF5")
    # ==================================================================
    # to_h5 writes files OsirisGridFile (and therefore Simulation) can read
    # again — the way to keep a derived quantity in the OSIRIS ecosystem.
    h5dir = args.outdir / "h5_export"
    d.name = "db3_dx1"
    d.to_h5(savename="db3_dx1", index=[0, 1, 2], path=str(h5dir))
    files = sorted(p.name for p in h5dir.glob("*.h5"))
    show("to_h5 wrote", files)
    back = ou.OsirisGridFile(str(h5dir / files[0]))
    show("read back: shape / units", (back.data.shape, back.units))
    show("values agree", np.allclose(back.data, d[0]))
    # d.to_h5(savename="db3_dx1", all=True)   # every frame, into <run>/MS/MISC/

    # ==================================================================
    section("7. Plain text and CSV")
    # ==================================================================
    profile = e1[3].mean(axis=1) if e1.dim > 1 else e1[3]
    ou.save_data(profile, str(args.outdir / "profile.txt"), option="numpy")
    ou.save_data(profile, str(args.outdir / "profile.csv"), option="pandas")
    show("np.savetxt round-trip", np.allclose(ou.read_data(str(args.outdir / "profile.txt")), profile))
    show("pandas round-trip", ou.read_data(str(args.outdir / "profile.csv"), option="pandas").shape)

    print("\nDone.")


if __name__ == "__main__":
    main()

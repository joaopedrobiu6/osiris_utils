"""Reading one OSIRIS file at a time: grid, RAW, tracks, HIST, TIMINGS.

:class:`osiris_utils.Simulation` is the right tool for a whole run.  The
readers here are the right tool for *one* file — a dump copied off a cluster, a
RAW snapshot, the energy history — and they are what ``Diagnostic`` uses
underneath.

All of them subclass :class:`osiris_utils.OsirisData`, which dispatches on the
filename: ``*.h5`` is HDF5, ``*_ene`` is a HIST table, ``timings*`` is a timing
report.

Run::

    python examples/scripts/02_single_file_readers.py [--sim DECK]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import load_simulation, parse_args, section, show

import osiris_utils as ou


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    run = Path(sim["e1"].simulation_folder)
    species = sim.species[0]

    # ------------------------------------------------------------------
    section("1. OsirisGridFile — one field dump")
    # ------------------------------------------------------------------
    path = sorted(Path(sim["e1"].path).glob("*.h5"))[3]
    grid = ou.OsirisGridFile(str(path))

    show("file", path.name)
    show("type / name", f"{grid.type} / {grid.name}")
    show("dim", grid.dim)
    show("nx", grid.nx)
    show("dx", grid.dx)
    show("grid bounds", grid.grid)
    show("data.shape", grid.data.shape)
    show("units / label", f"{grid.units} / {grid.label}")
    show("dt", grid.dt)
    show("iter", grid.iter)
    show("time [value, units]", grid.time)
    show("axis[0]['plot_label']", grid.axis[0]["plot_label"])
    # `verbose(True)` makes a reader narrate opening/closing its file.  It has
    # to be set before the file is opened, so it only helps on a subclass or on
    # OsirisData used directly — the constructors here open eagerly.
    show("np.asarray(grid)", np.asarray(grid).shape)  # __array__: use it like an array
    print(f"\n{grid}\n")  # __str__ prints the full metadata block

    # --- reading less than the whole file -----------------------------
    # data_slice goes into the HDF5 read: the rest never leaves the disk.
    # Slices are given in OSIRIS axis order (x1, x2, ...); the reader
    # transposes them for the stored layout.
    part = ou.OsirisGridFile(str(path), data_slice=(slice(10, 20),))
    show("data_slice=(10:20,)", part.data.shape)

    # Metadata only — no array read at all.  This is how Diagnostic learns the
    # grid without paying for a frame.
    meta = ou.OsirisGridFile(str(path), load_data=False)
    show("load_data=False -> data", meta.data)
    show("...but nx is there", meta.nx)

    # Array only — skips ~30 attribute reads, about a third of the cost of a
    # frame.  Used by Diagnostic._read_index, whose metadata is already known.
    fast = ou.OsirisGridFile(str(path), metadata=False)
    show("metadata=False -> data", fast.data.shape)
    show("...and no units", fast.units)

    # ------------------------------------------------------------------
    section("2. OsirisRawFile — a particle dump")
    # ------------------------------------------------------------------
    raw_dir = run / "MS" / "RAW" / species
    if not raw_dir.is_dir():
        show("no RAW dumps in this run", raw_dir)
    else:
        raw_path = sorted(raw_dir.glob("*.h5"))[-1]
        raw = ou.OsirisRawFile(str(raw_path))

        show("file", raw_path.name)
        show("species / type", f"{raw.name} / {raw.type}")
        show("time", raw.time)
        show("iter", raw.iter)
        show("grid bounds", raw.grid)
        show("quants", raw.quants)
        show("units['p1']", raw.units["p1"])
        show("labels['p1']", raw.labels["p1"])
        show("axis['x1']", raw.axis["x1"])
        show("data['x1'].shape", raw.data["x1"].shape)
        show("data['tag'].shape (node, id)", raw.data["tag"].shape)

        # Particle selection is plain numpy on the data dict.
        p1 = raw.data["p1"]
        fast_mask = p1 > np.percentile(p1, 90)
        show("particles with p1 in the top 10%", int(fast_mask.sum()))
        show("their mean energy", float(raw.data["ene"][fast_mask].mean()))

        # --- writing a tag file for the OSIRIS track diagnostic -------
        # 'all' takes every tag, 'random' samples n_tags of them, and `mask`
        # restricts the pool first.  Tags of already-tracked particles have a
        # negative node id; the writer takes abs() and sorts.
        tags_all = args.outdir / "tags_all.tags"
        tags_fast = args.outdir / "tags_fast.tags"
        args.outdir.mkdir(parents=True, exist_ok=True)
        raw.raw_to_file_tags(str(tags_all), type="all")
        raw.raw_to_file_tags(str(tags_fast), type="random", n_tags=16, mask=fast_mask)
        show("tag file head", tags_fast.read_text().splitlines()[:4])

        # The same writer, from any (n, 2) array of tags:
        ou.create_file_tags(str(args.outdir / "tags_manual.tags"), raw.data["tag"][:8])

    # ------------------------------------------------------------------
    section("3. OsirisTrackFile — tracked particles")
    # ------------------------------------------------------------------
    track_path = run / "MS" / "TRACKS" / f"{species}-tracks.h5"
    if not track_path.exists():
        show("no track file in this run", track_path)
    else:
        track = ou.OsirisTrackFile(str(track_path))
        show("num_particles", track.num_particles)
        show("num_time_iters", track.num_time_iters)
        show("quants", track.quants)
        show("units['x1']", track.units["x1"])
        # data is a structured array indexed [particle, time_index][quantity]
        show("data[0, :]['x1']", track.data[0, :]["x1"])
        show("data[:, -1]['ene'].shape", track.data[:, -1]["ene"].shape)
        show("time axis", track.time)

    # ------------------------------------------------------------------
    section("4. OsirisHIST — energy history (a '*_ene' text table)")
    # ------------------------------------------------------------------
    hist_files = sorted((run / "MS" / "HIST").glob("*_ene")) if (run / "MS" / "HIST").is_dir() else []
    if not hist_files:
        show("no HIST files in this run", run / "MS" / "HIST")
    else:
        hist = ou.OsirisHIST(str(hist_files[0]))
        show("columns", list(hist.df.columns))  # a pandas DataFrame
        print(hist.df.head().to_string(index=False))

    # ------------------------------------------------------------------
    section("5. OsirisTIMINGS — a timing report")
    # ------------------------------------------------------------------
    timing_files = sorted((run / "MS" / "TIMINGS").glob("timings*")) if (run / "MS" / "TIMINGS").is_dir() else []
    if not timing_files:
        show("no timing files in this run", run / "MS" / "TIMINGS")
    else:
        timings = ou.OsirisTIMINGS(str(timing_files[0]))
        show("iterations covered", timings.iterations)
        print(timings.df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

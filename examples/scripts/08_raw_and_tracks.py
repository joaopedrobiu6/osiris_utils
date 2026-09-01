"""Particles: RAW dumps, tag files, and the track diagnostic.

Two different things OSIRIS writes about particles:

* **RAW** — every particle of a species at *one* iteration.  Good for
  distribution functions and phase-space plots; no particle identity over time
  unless you match tags yourself.
* **TRACKS** — a chosen subset of particles at *every* iteration.  Good for
  trajectories and for following individual energisation.  Which particles get
  tracked is decided before the run by a ``file_tags`` file, which is normally
  generated from a RAW dump of an earlier run — that round trip is section 3.

Run::

    python examples/scripts/08_raw_and_tracks.py [--sim DECK] [--plot]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]
    run = Path(sim["e1"].simulation_folder)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    section("1. A RAW dump")
    # ==================================================================
    raw_files = sorted((run / "MS" / "RAW" / species).glob("*.h5")) if (run / "MS" / "RAW" / species).is_dir() else []
    if not raw_files:
        show("no RAW dumps in this run", run / "MS" / "RAW" / species)
        raw = None
    else:
        raw = ou.OsirisRawFile(str(raw_files[-1]))
        show("file", raw_files[-1].name)
        show("iteration / time", (raw.iter, raw.time[0]))
        show("quantities", raw.quants)
        show("particles", raw.data["x1"].shape[0])

        # data is a plain dict of arrays -> ordinary numpy from here on.
        p = np.stack([raw.data[q] for q in ("p1", "p2", "p3")])
        gamma = np.sqrt(1.0 + (p**2).sum(axis=0))
        show("<gamma> - 1", float((gamma - 1).mean()))
        show("gamma max", float(gamma.max()))

    # ==================================================================
    section("2. Selecting particles")
    # ==================================================================
    if raw is not None:
        x1 = raw.data["x1"]
        p1 = raw.data["p1"]

        # Any boolean mask: a region of the box, a momentum cut, a combination.
        front = 0.5 * (x1.min() + x1.max())
        downstream = x1 < front
        energetic = raw.data["ene"] > np.percentile(raw.data["ene"], 95)

        show("downstream particles", int(downstream.sum()))
        show("top 5% by energy", int(energetic.sum()))
        show("both", int((downstream & energetic).sum()))
        show("<p1> downstream vs upstream", (float(p1[downstream].mean()), float(p1[~downstream].mean())))

    # ==================================================================
    section("3. Writing a file_tags for the OSIRIS track diagnostic")
    # ==================================================================
    if raw is not None:
        # 'all' — track every particle in the dump (careful: that is a lot).
        raw.raw_to_file_tags(str(args.outdir / "tags_all.tags"), type="all")
        # 'random' — an unbiased sample of n_tags.
        raw.raw_to_file_tags(str(args.outdir / "tags_random.tags"), type="random", n_tags=32)
        # 'random' + mask — an unbiased sample *of a physically chosen subset*,
        # which is usually what you want: 16 of the most energetic particles.
        # n_tags may not exceed the size of the masked pool.
        tag_file = args.outdir / "tags_energetic.tags"
        raw.raw_to_file_tags(str(tag_file), type="random", n_tags=16, mask=energetic)
        try:
            raw.raw_to_file_tags(str(tag_file), type="random", n_tags=int(energetic.sum()) + 1, mask=energetic)
        except ValueError as e:
            show("asking for more than the mask holds", f"ValueError: {e}")

        print("\n".join(tag_file.read_text().splitlines()[:7]))

        # The low-level writer takes any (n, 2) array of (node, particle) tags.
        # It takes abs() of the node id, because a particle that is already
        # being tracked is stored with a negative one.
        ou.create_file_tags(str(args.outdir / "tags_manual.tags"), raw.data["tag"][:8])
        show("manual tag file", (args.outdir / "tags_manual.tags").exists())

        # Point the next run's deck at it:
        #     diag_species { file_tags = "tags_energetic.tags", ... }

    # ==================================================================
    section("4. Track_Diagnostic — trajectories from a run")
    # ==================================================================
    try:
        tracks = sim[species]["tracks"]
    except (FileNotFoundError, ValueError) as e:
        show("no tracks in this run", type(e).__name__)
        tracks = None

    if tracks is not None:
        show("species", tracks.species)
        show("num_particles", tracks.num_particles)
        show("num_time_iters", tracks.num_time_iters)
        show("quants", tracks.quants)
        show("dt / ndump", (tracks.dt, tracks.ndump))
        show("grid", tracks.grid)
        show("units['x1'] / labels['x1']", (tracks.units["x1"], tracks.labels["x1"]))
        show("dim", tracks.dim)
        show("path", tracks.path)

        # Indexing by quantity gives (n_particles, n_time) for that quantity.
        # Careful: until the file is loaded, each such access re-reads and
        # re-orders the whole track file — cheap once, wasteful in a loop.
        show("tracks['x1'].shape", tracks["x1"].shape)

        # load_all()/load() pull the whole structured array into memory; after
        # that, indexing slices it and `data`/`time` become available.
        tracks.load_all()
        show("after load_all: data", tracks.data.shape)
        show("time axis", tracks.time)
        show("data[particle, t]['ene']", tracks.data[0, -1]["ene"])
        tracks.unload()
        tracks.load()  # alias of load_all

        # Per-particle analysis is ordinary numpy.
        ene = tracks["ene"]
        gained = ene[:, -1] - ene[:, 0]
        show("most energised particle", int(np.argmax(gained)))
        show("its energy gain", float(gained.max()))

    # Tracks can also be opened file-by-file, without a Simulation:
    track_file = run / "MS" / "TRACKS" / f"{species}-tracks.h5"
    if track_file.exists():
        show("OsirisTrackFile directly", ou.OsirisTrackFile(str(track_file)).num_particles)

        # convert_tracks rewrites the modern 'tracks-2' layout into the older
        # one-group-per-particle format, which is easier to read by eye (and by
        # other tools).  It writes <name>-v2.h5 next to the input.
        copy = args.outdir / track_file.name
        copy.write_bytes(track_file.read_bytes())
        show("converted to", Path(ou.convert_tracks(str(copy))).name)

    # ==================================================================
    section("5. A plot")
    # ==================================================================
    if tracks is not None:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
        x1, ene = tracks["x1"], tracks["ene"]
        for i in range(min(6, tracks.num_particles)):
            ax1.plot(tracks.time, x1[i], lw=1)
            ax2.plot(tracks.time, ene[i], lw=1)
        ax1.set_xlabel(r"$t\ [1/\omega_p]$")
        ax1.set_ylabel(rf"${tracks.labels['x1']}$")
        ax2.set_xlabel(r"$t\ [1/\omega_p]$")
        ax2.set_ylabel(rf"${tracks.labels['ene']}$")
        fig.suptitle("tracked particles")
        savefig(fig, "08_tracks.png", args)
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

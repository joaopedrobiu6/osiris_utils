"""
create_databases.py — build one [t0, t1) database slice from an OSIRIS simulation.

Designed to be launched by the companion SLURM array job (create_db.sh).

Usage
-----
    python create_databases.py \\
        --sim-in  /path/to/shock_2D_elec_ion.in \\
        --outdir  /path/to/output \\
        --t0 800 --t1 1200 \\
        [--species electrons] \\
        [--workers 16] \\
        [--no-preload] \\
        [--profile]

Key optimisations vs. the naive version
----------------------------------------
* ``preload=True``  — all 1-D MFT averages are loaded into RAM once (in
  parallel) before the frame loop starts.  For a 2-D simulation this
  eliminates ~8 800 Lustre reads per 400-frame job and reduces per-frame
  work to pure NumPy.
* ``resume=True``   — a .progress checkpoint file lets the job pick up where
  it left off after a SLURM time-limit kill.
* ``max_workers``   — explicit thread count (match --cpus-per-task in the
  job script so SLURM accounting is honest).
* ``flush_every=50`` — checkpoint every 50 frames; low overhead for 400-frame
  jobs, protects against job preemption.
* ``OSIRIS_HDF5_CACHE_MB`` — env-var respected by OsirisGridFile._open_file_hdf5
  to size the per-file HDF5 chunk cache.  Set to 256 in the job script for a
  ~6× improvement in sequential read throughput on Lustre.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import osiris_utils as ou


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create one database slice [t0, t1) from an OSIRIS simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sim-in", required=True, help="Path to OSIRIS input deck (.in)")
    p.add_argument("--outdir", required=True, help="Output folder for .npy tensors")
    p.add_argument("--t0", type=int, required=True, help="Start dump index (inclusive)")
    p.add_argument("--t1", type=int, required=True, help="End dump index (exclusive)")
    p.add_argument("--species", default="electrons", help="Species name in the simulation")
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help=("ThreadPoolExecutor workers for parallel pre-loading and frame building. Match --cpus-per-task in your SLURM script."),
    )
    p.add_argument(
        "--no-preload",
        action="store_true",
        help="Disable MFT-average pre-loading (useful for debugging or tiny test slices).",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable osiris_utils profiling output (logs timing to stderr).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    if args.profile:
        ou.enable_profiling()

    # ── Sanity checks ────────────────────────────────────────────────────────
    if args.t0 >= args.t1:
        sys.exit(f"ERROR: t0={args.t0} must be < t1={args.t1}.")

    os.makedirs(args.outdir, exist_ok=True)

    # ── Unique output file names prevent collisions across array tasks ────────
    tag = f"{args.t0:04d}_{args.t1:04d}"
    name_input = f"input_tensor_{tag}"
    name_output = f"eta_tensor_{tag}"

    print(f"Slice [{args.t0}, {args.t1})  →  {args.outdir}/{name_input}.npy")
    print(f"  workers={args.workers}  preload={not args.no_preload}  resume=True")

    # ── Build ─────────────────────────────────────────────────────────────────
    sim = ou.Simulation(args.sim_in)
    ar_config = ou.AnomalousResistivityConfig(
        species="electrons",
        mft_axis=2,
        include_time_derivative=False,
        include_convection=True,
        include_transverse_advection=False,
        include_pressure=True,
        include_magnetic_force=True,
    )

    cfg = ou.DatabaseBuildConfig(
        # Parallel workers: pre-load phase and frame-building loop.
        max_workers=args.workers,
        # Pre-load 1-D MFT averages into RAM before the frame loop.
        # Eliminates ~8 800 Lustre reads for a 22-feature × 400-frame job.
        preload=not args.no_preload,
        ar_config=ar_config,
        # Resume an interrupted build — safe to re-run after SLURM timeout.
        resume=True,
        # Checkpoint every 50 frames (low overhead for 400-frame slices).
        flush_every=50,
    )

    db = ou.DatabaseCreator(
        simulation=sim,
        species=args.species,
        save_folder=args.outdir,
        build_config=cfg,
    )
    db.set_limits(initial_iter=args.t0, final_iter=args.t1)
    db.create_database(
        database="both",
        name_input=name_input,
        name_output=name_output,
    )


if __name__ == "__main__":
    main()

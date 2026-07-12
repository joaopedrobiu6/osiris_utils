"""
create_lorentz_database.py — build one Lorentz-augmented [t0, t1) database slice.

Each timestep is independently boosted by a random
:math:`\\beta_t \\sim \\mathcal{U}(\\beta_{\\min}, \\beta_{\\max})` before the
transverse average is taken.  The boost velocities are saved alongside the
feature tensor for reproducibility.

Designed to be launched by a companion SLURM array job.

Usage
-----
    python create_lorentz_database.py \\
        --sim-in  /path/to/shock_2D_elec_ion.in \\
        --outdir  /path/to/output \\
        --t0 800 --t1 1200 \\
        [--species electrons] \\
        [--workers 16] \\
        [--boost-min 0.0] \\
        [--boost-max 0.9] \\
        [--seed 42] \\
        [--database both] \\
        [--profile]

Output files (in --outdir)
--------------------------
    lorentz_tensor_<t0>_<t1>.npy        (T, 22, X)  float32 feature tensor
    lorentz_output_<t0>_<t1>.npy        (T,  1, X)  float32 eta tensor (boosted)
    boost_velocities_<t0>_<t1>.npy      (T,)         float32 beta values used

Key optimisations vs. the naive version
----------------------------------------
* ``resume=True``    — a .progress checkpoint file lets the job pick up where
  it left off after a SLURM time-limit kill, no frames wasted.
* ``max_workers``    — explicit thread count (match --cpus-per-task in the
  SLURM script so accounting is honest and h5py GIL-release parallelism is used).
* ``flush_every=50`` — checkpoint every 50 frames; low overhead for 400-frame
  jobs, protects against job preemption.
* ``seed``           — fixed seed makes the augmentation exactly reproducible
  across reruns and resumed jobs (boost_velocities is saved before any frames
  are written, so the same betas are used on resume).

Note on preload
---------------
``LorentzDatabaseBuildConfig`` does not expose a ``preload`` option.  Data is
always streamed frame-by-frame from disk, which is the only safe mode for 2-D
simulations.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import osiris_utils as ou


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create one Lorentz-augmented database slice [t0, t1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sim-in",  required=True, help="Path to OSIRIS input deck (.in)")
    p.add_argument("--outdir",  required=True, help="Output folder for .npy tensors")
    p.add_argument("--t0", type=int, required=True, help="Start dump index (inclusive)")
    p.add_argument("--t1", type=int, required=True, help="End dump index (exclusive)")
    p.add_argument("--species", default="electrons", help="Species name in the simulation")
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="ThreadPoolExecutor workers for parallel frame building. Match --cpus-per-task in your SLURM script.",
    )
    p.add_argument(
        "--boost-min",
        type=float,
        default=0.0,
        help="Lower bound of the uniform distribution for beta = v/c.",
    )
    p.add_argument(
        "--boost-max",
        type=float,
        default=0.9,
        help="Upper bound of the uniform distribution for beta = v/c. Must be < 1.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible boost sampling. Omit for a random draw each run.",
    )
    p.add_argument(
        "--database",
        default="both",
        choices=["input", "output", "both"],
        help=(
            "Which tensors to build. "
            "'input' = 22-feature boosted input tensor only; "
            "'output' = boosted eta (anomalous resistivity) only; "
            "'both' = input + output."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable osiris_utils profiling output (timing logged to stdout).",
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

    if not (0.0 <= args.boost_min < args.boost_max < 1.0):
        sys.exit(
            f"ERROR: boost values must satisfy 0 <= boost_min < boost_max < 1 "
            f"(got [{args.boost_min}, {args.boost_max}])."
        )

    os.makedirs(args.outdir, exist_ok=True)

    # ── Unique output file names prevent collisions across array tasks ────────
    tag = f"{args.t0:04d}_{args.t1:04d}"
    name_input  = f"lorentz_tensor_{tag}"
    name_output = f"lorentz_output_{tag}"
    name_boosts = f"boost_velocities_{tag}"

    print(f"Slice [{args.t0}, {args.t1})  database={args.database}  beta~U({args.boost_min}, {args.boost_max})  seed={args.seed}  →  {args.outdir}/")
    print(f"  workers={args.workers}  resume=True  flush_every=50")

    # ── Build ─────────────────────────────────────────────────────────────────
    sim = ou.Simulation(args.sim_in)

    cfg = ou.LorentzDatabaseBuildConfig(
        # Parallel workers — h5py releases the GIL so threads truly overlap I/O.
        max_workers=args.workers,
        boost_min=args.boost_min,
        boost_max=args.boost_max,
        seed=args.seed,
        # Resume an interrupted build — safe to re-run after SLURM timeout.
        resume=True,
        # Checkpoint every 50 frames (low overhead for 400-frame slices).
        flush_every=50,
    )

    db = ou.LorentzDatabaseCreator(
        simulation=sim,
        species=args.species,
        save_folder=args.outdir,
        build_config=cfg,
    )
    db.set_limits(initial_iter=args.t0, final_iter=args.t1)
    db.create_database(
        database=args.database,
        name_input=name_input,
        name_output=name_output,
        name_boosts=name_boosts,
    )


if __name__ == "__main__":
    main()

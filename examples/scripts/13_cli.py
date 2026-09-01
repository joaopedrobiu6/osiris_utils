"""The `utils` command-line tool, driven from Python so it is runnable here.

Installing the package puts a ``utils`` command on your PATH::

    utils info      <deck|file|dir>   metadata about a run or a single file
    utils validate  <deck|file|dir>   integrity checks, optionally for gaps
    utils export    <file|dir>        one file (or one diagnostic) to csv/json/npy
    utils plot      <file>            a quick PNG from one dump

They exist for the cluster: checking what a job produced, spotting a missing
dump, or eyeballing one file without opening Python.  Everything they do is
also available from the API — ``utils export`` is for a single file, while
``ou.export_to_npy`` is the one that scales to a whole time series (example 09).

This script shells out to ``python -m osiris_utils.cli`` so it works from a
checkout too.  Add ``-v`` to any command to get the full traceback on failure.

Run::

    python examples/scripts/13_cli.py [--sim DECK]
"""

from __future__ import annotations

import subprocess
import sys

from _common import deck_path, parse_args, section, show


def run(*argv: str) -> None:
    """Run one CLI command and echo it with its output."""
    cmd = [sys.executable, "-m", "osiris_utils.cli", *argv]
    print(f"\n$ utils {' '.join(argv)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    body = (proc.stdout + proc.stderr).strip()
    # Drop the runpy warning that only appears because we use -m here.
    body = "\n".join(line for line in body.splitlines() if "RuntimeWarning" not in line and "found in sys.modules" not in line)
    print("\n".join(f"    {line}" for line in body.splitlines()[:22]))
    if proc.returncode:
        print(f"    [exit {proc.returncode}]")


def main() -> None:
    args = parse_args(__doc__)
    deck = deck_path(args)
    run_dir = deck.parent
    args.outdir.mkdir(parents=True, exist_ok=True)

    one_file = sorted((run_dir / "MS" / "FLD" / "e1").glob("*.h5"))[3]

    section("1. utils info — what is in this run?")
    run("info", str(deck))
    run("info", str(deck), "--brief")
    run("info", str(one_file))
    # A directory works too: the deck is looked up by name (os-stdin,
    # input.deck, deck.in).  A deck named anything else must be passed directly.
    run("info", str(run_dir))

    section("2. utils validate — is anything missing or corrupt?")
    run("validate", str(one_file))
    run("validate", str(deck))
    # --check-missing looks for gaps in each diagnostic's dump sequence;
    # --strict turns warnings into a non-zero exit, for a CI or job script.
    run("validate", str(deck), "--check-missing")
    run("validate", str(deck), "--strict")

    section("3. utils export — one file to csv / json / npy")
    run("export", str(one_file), "--format", "csv", "--output", str(args.outdir / "e1.csv"))
    run("export", str(one_file), "--format", "json", "--output", str(args.outdir / "e1.json"))
    run("export", str(one_file), "--format", "npy", "--output", str(args.outdir / "e1.npy"))
    # --no-coords writes the data alone; -t picks one timestep out of a
    # diagnostic *directory* (a single file is already one timestep).
    run("export", str(one_file), "--format", "csv", "--no-coords", "--output", str(args.outdir / "e1_bare.csv"))
    run("export", str(run_dir / "MS" / "FLD" / "e1"), "--format", "npy", "-t", "2", "--output", str(args.outdir / "e1_t2.npy"))

    section("4. utils plot — a PNG straight from a dump")
    run("plot", str(one_file), "--save", str(args.outdir / "cli_e1.png"))
    run(
        "plot",
        str(one_file),
        "--save",
        str(args.outdir / "cli_e1_viridis.png"),
        "--cmap",
        "viridis",
        "--title",
        "E_1 at dump 3",
        "--dpi",
        "120",
    )
    # --log-scale for data spanning orders of magnitude; --display opens a
    # window instead of writing a file (needs an X server / GUI backend).
    run("plot", str(one_file), "--save", str(args.outdir / "cli_e1_log.png"), "--log-scale")

    section("5. Housekeeping")
    run("--version")
    run("--help")

    show("files written", sorted(p.name for p in args.outdir.glob("*e1*")))
    print("\nDone.")


if __name__ == "__main__":
    main()

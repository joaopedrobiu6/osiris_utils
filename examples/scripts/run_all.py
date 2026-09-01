"""Run every example script in order and report which ones passed.

Useful as a smoke test after changing the package: the examples touch nearly
every public entry point, so a break shows up here immediately.

Run::

    python examples/scripts/run_all.py            # synthetic run, no figures
    python examples/scripts/run_all.py --plot     # also write the PNGs
    python examples/scripts/run_all.py --sim RUN/os-stdin   # against real data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", default=None, help="input deck to run the examples against")
    parser.add_argument("--plot", action="store_true", help="pass --plot to every example")
    parser.add_argument("--outdir", default=None, help="pass --outdir to every example")
    args = parser.parse_args()

    scripts = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))
    extra: list[str] = []
    if args.sim:
        extra += ["--sim", args.sim]
    if args.plot:
        extra += ["--plot"]
    if args.outdir:
        extra += ["--outdir", args.outdir]

    failures = []
    for script in scripts:
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script), *extra], capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        status = "ok  " if proc.returncode == 0 else "FAIL"
        print(f"{status} {script.name:<44} {elapsed:6.1f}s")
        if proc.returncode != 0:
            failures.append(script.name)
            print("\n".join(f"       {line}" for line in proc.stderr.strip().splitlines()[-12:]))

    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} examples passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

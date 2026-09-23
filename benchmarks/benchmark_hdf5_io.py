"""Cost of opening and reading one OSIRIS grid dump, with and without its metadata."""

import time

import numpy as np
from _synthetic import field_series

import osiris_utils as ou

SHAPE = (1024, 256)  # grid points per frame
N_READS = 100


def read_times(metadata: bool) -> np.ndarray:
    path = next((field_series("e3", 4, SHAPE) / "MS" / "FLD" / "e3").glob("*.h5"))
    ou.OsirisGridFile(str(path), metadata=metadata)  # warm-up
    times = np.empty(N_READS)
    for i in range(N_READS):
        start = time.perf_counter()
        ou.OsirisGridFile(str(path), metadata=metadata)
        times[i] = time.perf_counter() - start
    return times


if __name__ == "__main__":
    size_mb = np.prod(SHAPE) * 4 / 1024**2
    for metadata in (True, False):
        t = read_times(metadata)
        print(f"metadata={metadata!s:5}  mean {t.mean() * 1e3:6.2f} ms  min {t.min() * 1e3:6.2f} ms  ({size_mb / t.mean():6.0f} MB/s)")

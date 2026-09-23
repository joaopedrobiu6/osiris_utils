"""Sequential vs threaded Diagnostic.load_all() on a synthetic 2-D field series."""

import time

from _synthetic import field_series

import osiris_utils as ou

NT = 128  # frames
SHAPE = (1024, 256)  # grid points per frame (float32: 1 MB per frame)


def load_time(use_parallel: bool) -> tuple[float, tuple]:
    d = ou.Diagnostic(simulation_folder=str(field_series("e3", NT, SHAPE)))
    d.get_quantity("e3")
    start = time.perf_counter()
    data = d.load_all(use_parallel=use_parallel)
    return time.perf_counter() - start, data.shape


if __name__ == "__main__":
    load_time(use_parallel=False)  # warm the OS page cache so both runs read from memory
    t_seq, shape = load_time(use_parallel=False)
    t_par, _ = load_time(use_parallel=True)
    print(f"load_all of {shape} float32")
    print(f"  sequential : {t_seq:.3f} s")
    print(f"  threads    : {t_par:.3f} s   ({t_seq / t_par:.2f}x)")

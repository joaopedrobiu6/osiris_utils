"""Read times of RAW and track files: whole file, a subset of quantities, and the track index build."""

import time

import h5py
from _synthetic import raw_dump, track_file

import osiris_utils as ou
from osiris_utils.data.data import get_track_indexes
from osiris_utils.data.track_diagnostic import Track_Diagnostic

N_REPEATS = 5


def best_time(fn) -> float:
    fn()  # warm-up, and the OS page cache
    times = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


if __name__ == "__main__":
    raw, track = raw_dump(), track_file()
    with h5py.File(track) as f:
        itermap, n_particles = f["itermap"][()], int(f.attrs["NTRACKS"][0])
    tracks = Track_Diagnostic(str(track.parents[2]), species=ou.Species("electrons", -1))

    print(f"RAW   {raw.stat().st_size / 1e6:5.0f} MB    track {track.stat().st_size / 1e6:5.0f} MB  (best of {N_REPEATS}, warm page cache)")
    rows = [
        ("RAW, all quantities", lambda: ou.OsirisRawFile(str(raw))),
        ("RAW, quants=['x1', 'p1']", lambda: ou.OsirisRawFile(str(raw), quants=["x1", "p1"])),
        ("track index build", lambda: get_track_indexes(itermap, n_particles)),
        ("track, all quantities", lambda: ou.OsirisTrackFile(str(track))),
        ("Track_Diagnostic['x1'], not loaded", lambda: tracks["x1"]),
    ]
    for label, fn in rows:
        print(f"  {label:38s} {best_time(fn) * 1e3:7.1f} ms")

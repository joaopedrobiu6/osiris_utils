import time

import osiris_utils as ou


def benchmark_hdf5_io():
    """Benchmark HDF5 file opening and reading."""
    print("=" * 60)
    print("Benchmarking HDF5 I/O Performance")
    print("=" * 60)

    filepath = "examples/example_data/MS/FLD/e3/e3-000100.h5"

    # Warm-up (file may be cached by OS)
    _ = ou.OsirisGridFile(filepath)

    # Benchmark multiple loads
    n_iterations = 100
    times = []

    print(f"\nLoading file {n_iterations} times...")
    for _ in range(n_iterations):
        start = time.time()
        data = ou.OsirisGridFile(filepath)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\nResults:")
    print(f"  Average load time: {avg_time * 1000:.2f}ms")
    print(f"  Min load time: {min_time * 1000:.2f}ms")
    print(f"  Max load time: {max_time * 1000:.2f}ms")
    print(f"  Data shape: {data.data.shape}")
    print(f"  Data size: {data.data.nbytes / 1024:.1f} KB")

    # Calculate throughput
    throughput = (data.data.nbytes / 1024 / 1024) / avg_time
    print(f"  Throughput: {throughput:.1f} MB/s")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_hdf5_io()

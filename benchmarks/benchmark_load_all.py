import sys
import time

import osiris_utils as ou

# Use example data if available
data_path = "examples/example_data"


def benchmark_loading():
    """Benchmark parallel vs sequential loading."""
    _ = f"{data_path}/MS/FLD/e3"

    print("=" * 60)
    print("Benchmarking Diagnostic.load_all() performance")
    print("=" * 60)

    # Test with sequential loading
    print("\n1. Testing SEQUENTIAL loading...")
    d = ou.Diagnostic(simulation_folder=data_path)
    d.get_quantity("e3")

    start = time.time()
    data_seq = d.load_all(use_parallel=False)
    seq_time = time.time() - start
    print(f"   Sequential time: {seq_time:.3f}s for {len(d)} timesteps")
    print(f"   Data shape: {data_seq.shape}")

    # Unload to test parallel
    d.unload()

    # Test with parallel loading
    print("\n2. Testing PARALLEL loading...")
    start = time.time()
    data_par = d.load_all(use_parallel=True)
    par_time = time.time() - start
    print(f"   Parallel time: {par_time:.3f}s for {len(d)} timesteps")
    print(f"   Data shape: {data_par.shape}")

    # Calculate speedup
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\n✓ Speedup: {speedup:.2f}x faster with parallel loading")
    print(f"  Time saved: {seq_time - par_time:.3f}s ({100 * (seq_time - par_time) / seq_time:.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        benchmark_loading()
    except Exception as e:
        print(f"Error running benchmark: {e}")
        print("Make sure example_data is available in the examples directory")
        sys.exit(1)

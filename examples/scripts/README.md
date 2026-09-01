# Example scripts

Runnable, annotated tours of every public part of `osiris_utils`. Each script
covers one area, prints what it computes, and where a result has a closed form
it checks it — so the output is evidence, not decoration.

## Running them

```bash
pip install -e .                       # from a checkout, once
python examples/scripts/run_all.py     # every example, ~35 s
```

Or one at a time:

```bash
python examples/scripts/04_derivatives.py
```

Every script takes the same options:

| option | meaning |
|---|---|
| `--sim DECK` | run against your own OSIRIS run (pass its **input deck**) |
| `--plot` | also write figures to `--outdir` |
| `--outdir DIR` | where figures and output files go (default `examples/scripts/output/`) |
| `--log` | keep the package's INFO logging, which is silenced by default |

**No data is needed.** With no `--sim`, the scripts build a small synthetic
OSIRIS run (~4 MB, cached under the OS temp directory) whose HDF5 layout and
schema are the ones OSIRIS writes, so the package reads it exactly as it reads a
real run. See [`synthetic_run.py`](synthetic_run.py) for what it contains and
the analytic profile behind it — a shock-like ramp in x1, modulated
periodically in x2, advected in time.

```bash
python examples/scripts/synthetic_run.py --root /tmp/demo_run   # write it yourself
```

## The scripts

| script | covers |
|---|---|
| [`01_simulation_and_diagnostics.py`](01_simulation_and_diagnostics.py) | `Simulation`, `Species_Handler`, `Diagnostic`: metadata, lazy frames, spatial slicing, `load_all` (threads/processes), arithmetic, custom diagnostics, `to_h5` |
| [`02_single_file_readers.py`](02_single_file_readers.py) | `OsirisGridFile` (slicing, `load_data`, `metadata`), `OsirisRawFile`, `OsirisTrackFile`, `OsirisHIST`, `OsirisTIMINGS` |
| [`03_input_decks.py`](03_input_decks.py) | `InputDeckIO`: parsing, `get_param`/`set_param`/`set_tag`/`delete_param`, templating a scan, writing a deck back |
| [`04_derivatives.py`](04_derivatives.py) | `Derivative_Simulation` / `Derivative_Diagnostic`: every `deriv_type`, order 2/4, custom stencils, higher `deriv_order`, periodic boundaries, chaining, accuracy against an exact derivative |
| [`05_fft.py`](05_fft.py) | `FFT_Simulation` / `FFT_Diagnostic`: spatial and space–time transforms, windowing, detrending, normalisation, `k()` / `omega()` |
| [`06_mft_and_filters.py`](06_mft_and_filters.py) | `MFT_*` (`avg` / `delta`), the `SpatialFilter` family, `FilterChain`, `as_filter`, `Filtered_Simulation`, and filtered derivatives |
| [`07_field_centering_and_corrections.py`](07_field_centering_and_corrections.py) | Yee-mesh centering, pressure correction (`P - n u v`), heat-flux correction |
| [`08_raw_and_tracks.py`](08_raw_and_tracks.py) | RAW dumps, particle selection, writing `file_tags`, `Track_Diagnostic`, `convert_tracks` |
| [`09_export_and_io.py`](09_export_and_io.py) | `export_to_npy` (reductions, time averaging, workers, checkpointing), `export_simulation_to_npy`, HDF5 round-trips, text I/O |
| [`10_anomalous_resistivity.py`](10_anomalous_resistivity.py) | `AnomalousResistivity`: `e_vlasov`, `LHS`, `eta` / `eta_new`, every term and its coefficient, physics flags, other species, per-frame filtering |
| [`11_databases.py`](11_databases.py) | `DatabaseCreator` (all tensor types and build options), burst dumps + `BurstAxis`, `LorentzDatabaseCreator` |
| [`12_utils_profiling_vis.py`](12_utils_profiling_vis.py) | run planning (`courant2D`, `time_estimation`, `filesize_estimation`), `integrate`, `transverse_average`, profiling, `plot_3d`, frame-cache tuning |
| [`13_cli.py`](13_cli.py) | the `utils` command: `info`, `validate`, `export`, `plot` |

## Reading order

Start at 01 — everything else builds on `Diagnostic` and its laziness. Then:

- **analysing fields**: 04 → 05 → 06
- **particles**: 02 → 08
- **the anomalous-resistivity pipeline**: 06 → 10 → 11
- **getting data out**: 09, 13

## Notes

- Two things stay lazy throughout: a `Diagnostic` reads one dump on demand, and
  arithmetic or post-processing on it returns another lazy `Diagnostic`. Nothing
  calls `load_all()` for you, because a 3-D run does not fit in memory.
- The transverse axis (x2) is periodic and the longitudinal one (x1) is open
  everywhere in these examples — the shock-simulation convention the filters and
  the databases assume. Pass `periodic=` / `periodic_axes=` if your run differs.
- Figures are only written with `--plot`, so the scripts run headless in a batch
  job without a display.

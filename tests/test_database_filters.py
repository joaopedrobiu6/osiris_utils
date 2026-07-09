"""Tests for the database spatial filters and the per-frame tensor pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from osiris_utils.ar import AnomalousResistivityConfig
from osiris_utils.database.database import (
    INPUT_FEATURE_LABELS,
    OUTPUT_LABELS,
    VNT_BASE_QUANTITIES,
    VNT_DERIV_ORDERS,
    DatabaseBuildConfig,
    DatabaseCreator,
    _mean_field_frame_quantities,
    _stack_rows,
    _vnT_frame_quantities,
    vnT_feature_labels,
)
from osiris_utils.database.filters import (
    FilterChain,
    GaussianFilter,
    NoFilter,
    SavitzkyGolayFilter,
    as_filter,
    fd_derivative,
)
from osiris_utils.database.lorentz_database import (
    LORENTZ_FEATURE_LABELS,
    LorentzDatabaseBuildConfig,
    LorentzDatabaseCreator,
    _boost_combined_frame,
)

# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------


def test_fd_derivative_sine_nonperiodic():
    x = np.linspace(0.0, 2.0 * np.pi, 200)
    dx = x[1] - x[0]
    d = fd_derivative(np.sin(x), dx, axis=0, order=1, periodic=False)
    assert np.allclose(d, np.cos(x), atol=1e-4)


def test_fd_derivative_sine_periodic():
    x = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    dx = x[1] - x[0]
    d = fd_derivative(np.sin(x), dx, axis=0, order=1, periodic=True)
    assert np.allclose(d, np.cos(x), atol=1e-5)


def test_fd_derivative_second_order_is_repeated_first():
    x = np.linspace(0.0, 2.0 * np.pi, 200)
    dx = x[1] - x[0]
    f = np.sin(x)
    d2 = fd_derivative(f, dx, order=2)
    assert np.allclose(d2, fd_derivative(fd_derivative(f, dx), dx))
    # interior should approximate -sin
    assert np.allclose(d2[10:-10], -np.sin(x)[10:-10], atol=1e-3)


def test_fd_derivative_2d_axis():
    x = np.linspace(0.0, 2.0 * np.pi, 100)
    y = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.sin(X) * np.cos(Y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    assert np.allclose(fd_derivative(f, dx, axis=0), np.cos(X) * np.cos(Y), atol=1e-3)
    assert np.allclose(fd_derivative(f, dy, axis=1, periodic=True), -np.sin(X) * np.sin(Y), atol=1e-3)


def test_no_filter_is_identity_with_fd_derivatives():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(32, 16))
    filt = NoFilter()
    assert np.array_equal(filt.smooth(f, periodic=(False, True)), f)
    assert np.array_equal(filt.derivative(f, 0.1, axis=0, order=2), fd_derivative(f, 0.1, axis=0, order=2))


def test_savgol_smooth_and_derivative_exact_on_cubic():
    # A polynomial of degree <= polyorder is reproduced exactly by SG,
    # and its SG derivatives equal the analytic ones.
    x = np.linspace(-1.0, 1.0, 51)
    dx = x[1] - x[0]
    f = 2.0 + x - 0.5 * x**2 + 0.25 * x**3
    filt = SavitzkyGolayFilter(window_length=9, polyorder=3)
    assert np.allclose(filt.smooth(f), f, atol=1e-12)
    assert np.allclose(filt.derivative(f, dx, order=1), 1.0 - x + 0.75 * x**2, atol=1e-10)
    assert np.allclose(filt.derivative(f, dx, order=2), -1.0 + 1.5 * x, atol=1e-9)


def test_savgol_periodic_derivative():
    x = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    dx = x[1] - x[0]
    filt = SavitzkyGolayFilter(window_length=7, polyorder=3)
    d = filt.derivative(np.sin(x), dx, order=1, periodic=True)
    assert np.allclose(d, np.cos(x), atol=1e-3)


def test_savgol_validation():
    with pytest.raises(ValueError):
        SavitzkyGolayFilter(window_length=6, polyorder=2)  # even window
    with pytest.raises(ValueError):
        SavitzkyGolayFilter(window_length=5, polyorder=5)  # polyorder >= window
    # polyorder 0 can smooth (moving average) but has no derivative operator
    filt = SavitzkyGolayFilter(window_length=7, polyorder=0)
    with pytest.raises(ValueError, match="polyorder"):
        filt.derivative(np.zeros(32), 0.1, order=1)


def test_derivative_order_beyond_polyorder_raises():
    # Single-pass derivatives need the local fit to support the order —
    # silent chaining (extra smoothing per order) is no longer available.
    filt = SavitzkyGolayFilter(window_length=7, polyorder=2)
    with pytest.raises(ValueError, match="polyorder"):
        filt.derivative(np.sin(np.linspace(0, 6, 64)), 0.1, order=4, periodic=True)


def test_nofilter_derivative_order_is_repeated_first():
    # NoFilter has no smoothing kernel: order=n IS n chained first derivatives.
    rng = np.random.default_rng(4)
    f = rng.normal(size=(48, 16))
    filt = NoFilter()
    d1 = filt.derivative(f, 0.1, axis=0, order=1)
    assert np.allclose(filt.derivative(f, 0.1, axis=0, order=2), filt.derivative(d1, 0.1, axis=0, order=1))
    assert np.allclose(filt.derivative(f, 0.1, axis=0, order=3), filt.derivative(d1, 0.1, axis=0, order=2))


@pytest.mark.parametrize(
    "filt",
    [
        SavitzkyGolayFilter(window_length=7, polyorder=4),
        GaussianFilter(sigma=1.5),
        FilterChain(GaussianFilter(sigma=1.0), SavitzkyGolayFilter(window_length=7, polyorder=4)),
    ],
    ids=["savgol", "gaussian", "chain"],
)
def test_smoothing_derivatives_are_single_pass(filt):
    rng = np.random.default_rng(4)
    f = rng.normal(size=(48, 16))
    # order=1: identical to the old (chained) implementation — one pass either way.
    d1 = filt.derivative(f, 0.1, axis=0, order=1)
    assert np.array_equal(d1, filt._first_derivative(np.asarray(f, dtype=np.float64), 0.1, axis=0, periodic=False))
    # order=2: a single kernel pass, NOT two chained first derivatives
    # (chaining smooths twice → visibly more attenuated on noise).
    d2_native = filt.derivative(f, 0.1, axis=0, order=2)
    d2_chained = filt.derivative(d1, 0.1, axis=0, order=1)
    assert not np.allclose(d2_native, d2_chained)


def test_gaussian_high_order_attenuation_matches_single_pass():
    # On sin(kx), one Gaussian pass attenuates by exp(-(k*sigma_phys)^2 / 2).
    # Native d2 must show ONE pass of attenuation; the old chained scheme
    # carried two (effective sigma*sqrt(2)) — kept here as the reference for
    # what the fix removed.
    n_pts = 256
    x = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    dx = x[1] - x[0]
    k, sigma_pts = 4.0, 4.0
    sigma_phys = sigma_pts * dx
    f = np.sin(k * x)
    filt = GaussianFilter(sigma=sigma_pts)
    atten1 = np.exp(-0.5 * (k * sigma_phys) ** 2)

    d2_native = filt.derivative(f, dx, order=2, periodic=True)
    assert np.allclose(d2_native, -(k**2) * atten1 * f, rtol=5e-3, atol=5e-3 * k**2)

    d1 = filt.derivative(f, dx, order=1, periodic=True)
    d2_chained = filt.derivative(d1, dx, order=1, periodic=True)
    assert np.allclose(d2_chained, -(k**2) * atten1**2 * f, rtol=5e-3, atol=5e-3 * k**2)
    # the two schemes differ by exactly one extra smoothing pass
    assert not np.allclose(d2_native, d2_chained, rtol=1e-2)


def test_gaussian_high_order_derivative_accuracy():
    # d3/d4 kernels are internally widened (truncate >= 8 sigma): with the
    # smoothing default truncate=4 the kernel truncation tails dominate the
    # ~k^order low-k response and d4 is orders of magnitude wrong.
    n_pts = 512
    x = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    dx = x[1] - x[0]
    k, sigma_pts = 3.0, 3.0
    f = np.sin(k * x)
    filt = GaussianFilter(sigma=sigma_pts)
    atten1 = np.exp(-0.5 * (k * sigma_pts * dx) ** 2)

    d3 = filt.derivative(f, dx, order=3, periodic=True)
    assert np.allclose(d3, -(k**3) * atten1 * np.cos(k * x), rtol=1e-6, atol=1e-6 * k**3)
    d4 = filt.derivative(f, dx, order=4, periodic=True)
    assert np.allclose(d4, k**4 * atten1 * f, rtol=1e-6, atol=1e-6 * k**4)


def test_savgol_axes_restriction():
    # axes=(0,) must leave variation along axis 1 untouched.
    rng = np.random.default_rng(1)
    f = np.tile(rng.normal(size=(1, 16)), (32, 1))  # constant along axis 0
    filt = SavitzkyGolayFilter(window_length=5, polyorder=2, axes=(0,))
    assert np.allclose(filt.smooth(f), f, atol=1e-12)


def test_gaussian_smooth_and_derivative_sine():
    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    dx = x[1] - x[0]
    sigma_pts = 2.0
    filt = GaussianFilter(sigma=sigma_pts)
    # Gaussian smoothing of sin(kx) attenuates by exp(-(k*sigma_phys)^2 / 2)
    atten = np.exp(-0.5 * (sigma_pts * dx) ** 2)
    assert np.allclose(filt.smooth(np.sin(x), periodic=True), atten * np.sin(x), rtol=1e-3, atol=1e-4)
    d = filt.derivative(np.sin(x), dx, order=1, periodic=True)
    assert np.allclose(d, atten * np.cos(x), rtol=1e-3, atol=1e-4)


def test_gaussian_reduces_noise_variance():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(64, 64))
    filt = GaussianFilter(sigma=2.0)
    assert filt.smooth(f, periodic=(True, True)).std() < 0.5 * f.std()


def test_filter_chain_smooth_and_derivative():
    g1 = GaussianFilter(sigma=1.0)
    g2 = SavitzkyGolayFilter(window_length=5, polyorder=2)
    chain = FilterChain(g1, g2)
    rng = np.random.default_rng(3)
    f = rng.normal(size=64)
    assert np.allclose(chain.smooth(f), g2.smooth(g1.smooth(f)))
    # derivative delegates to the last filter's scheme
    assert np.allclose(chain.derivative(f, 0.1), g2.derivative(f, 0.1))


def test_as_filter():
    assert isinstance(as_filter(None), NoFilter)
    assert isinstance(as_filter(()), NoFilter)
    g = GaussianFilter(sigma=1.0)
    assert as_filter(g) is g
    assert as_filter([g]) is g
    chain = as_filter([g, NoFilter()])
    assert isinstance(chain, FilterChain)
    with pytest.raises(TypeError):
        FilterChain(g, "not a filter")
    with pytest.raises(ValueError):
        FilterChain()


def test_periodic_length_mismatch_raises():
    filt = GaussianFilter(sigma=1.0)
    with pytest.raises(ValueError, match="periodic"):
        filt.smooth(np.zeros((8, 8)), periodic=(True,))


# ----------------------------------------------------------------------
# Synthetic fields + mock simulation
# ----------------------------------------------------------------------

NX, NY, NT = 48, 16, 4


def _make_fields(y_dep: bool = True) -> tuple[dict[str, np.ndarray], float, float]:
    """Smooth analytic (T, nx, ny) fields; y-dependence optional."""
    x = np.linspace(0.0, 2.0 * np.pi, NX)
    y = np.linspace(0.0, 2.0 * np.pi, NY, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = float(x[1] - x[0])
    dx2 = float(y[1] - y[0])

    def field(base: float, amp: float, y_amp: float, phase: float) -> np.ndarray:
        frames = []
        for t in range(NT):
            fr = base + amp * np.sin(X + phase + 0.1 * t)
            if y_dep:
                fr = fr + y_amp * np.cos(2.0 * Y + phase)
            frames.append(fr)
        return np.stack(frames)

    fields = {
        "n": field(1.5, 0.2, 0.05, 0.0),
        "T11": field(1.0, 0.1, 0.03, 0.3),
        "T12": field(0.1, 0.05, 0.02, 0.7),
        "vfl1": field(0.2, 0.1, 0.03, 1.1),
        "vfl2": field(0.05, 0.04, 0.02, 1.5),
        "vfl3": field(0.02, 0.03, 0.01, 1.9),
        "b2": field(0.1, 0.05, 0.02, 2.3),
        "b3": field(0.3, 0.1, 0.03, 2.7),
        # extra fields for the Lorentz pipeline
        "e1": field(0.05, 0.02, 0.01, 3.1),
        "e2": field(0.05, 0.02, 0.01, 3.1),
        "e3": field(0.04, 0.02, 0.01, 3.5),
        "P11": field(1.6, 0.15, 0.04, 0.5),
        "P12": field(0.15, 0.05, 0.02, 0.9),
        "P00": field(2.0, 0.2, 0.05, 1.3),
        "ufl1": field(0.2, 0.1, 0.03, 1.1),
        "ufl2": field(0.05, 0.04, 0.02, 1.5),
    }
    return fields, dx, dx2


class MockDiagnostic:
    def __init__(self, data: np.ndarray, dx: tuple[float, float]):
        self._data = data
        self.dx = dx
        self.nx = data.shape[1:]
        self.x = (np.arange(data.shape[1]) * dx[0], np.arange(data.shape[2]) * dx[1])

    def __getitem__(self, t_idx: int) -> np.ndarray:
        return self._data[t_idx]

    def __len__(self) -> int:
        return self._data.shape[0]


class MockSpecies:
    def __init__(self, diags: dict[str, MockDiagnostic]):
        self._diags = diags

    def __getitem__(self, key: str) -> MockDiagnostic:
        return self._diags[key]


class MockSimulation:
    def __init__(self, fields: dict[str, np.ndarray], dx: tuple[float, float], species: str = "electrons"):
        diags = {name: MockDiagnostic(data, dx) for name, data in fields.items()}
        self._field_diags = diags
        self._species_handler = MockSpecies(diags)
        self._species_name = species

    def __getitem__(self, key: str):
        if key == self._species_name:
            return self._species_handler
        return self._field_diags[key]


def _default_flags() -> AnomalousResistivityConfig:
    return AnomalousResistivityConfig(species="electrons")


# ----------------------------------------------------------------------
# Mean-field frame pipeline
# ----------------------------------------------------------------------


def test_mean_field_frame_features_match_manual_fd():
    fields, dx, dx2 = _make_fields()
    q = _mean_field_frame_quantities(fields, 0, NoFilter(), dx, dx2, avg_axis=1, flags=_default_flags())

    n_avg = fields["n"][0].mean(axis=1)
    T11_avg = fields["T11"][0].mean(axis=1)

    vfl1_avg = fields["vfl1"][0].mean(axis=1)

    assert np.allclose(q["n_avg"], n_avg)
    # Per-field derivatives: taken on the 2-D data (x = axis 0), then averaged
    assert np.allclose(q["d1_vfl1_dx1_avg"], fd_derivative(fields["vfl1"][0], dx, axis=0).mean(axis=1))
    assert np.allclose(q["d2_n_dx1_avg"], fd_derivative(fields["n"][0], dx, axis=0, order=2).mean(axis=1))
    assert np.allclose(q["d4_T12_dx1_avg"], fd_derivative(fields["T12"][0], dx, axis=0, order=4).mean(axis=1))
    # Mean-field composites: derivative of the product of the averaged profiles
    assert np.allclose(q["d1_nT11_dx1_avg"], fd_derivative(n_avg * T11_avg, dx))
    assert np.allclose(q["nvfl1_avg"], n_avg * vfl1_avg)
    assert np.allclose(q["d1_nvfl1_dx1_avg"], fd_derivative(n_avg * vfl1_avg, dx))
    assert np.allclose(q["dnT11_dx1_over_n_avg"], q["d1_nT11_dx1_avg"] / n_avg)

    frame = _stack_rows(q, INPUT_FEATURE_LABELS)
    assert frame.shape == (len(INPUT_FEATURE_LABELS), NX)
    assert np.all(np.isfinite(frame))


def test_stack_rows_reports_missing_label():
    with pytest.raises(KeyError, match="not computed"):
        _stack_rows({"a": np.zeros(4)}, ["a", "missing_label"])


@pytest.mark.parametrize("eta_formula", ["thesis", "lhs"])
@pytest.mark.parametrize(
    "filt",
    [NoFilter(), SavitzkyGolayFilter(window_length=7, polyorder=4)],
    ids=["nofilter", "savgol"],
)
def test_eta_vanishes_for_y_independent_fields(eta_formula, filt):
    # With no transverse variation, all fluctuations vanish (thesis eta = 0)
    # and the mean-field corrections cancel e_vlasov exactly (lhs eta = 0).
    fields, dx, dx2 = _make_fields(y_dep=False)
    q = _mean_field_frame_quantities(fields, 0, filt, dx, dx2, avg_axis=1, flags=_default_flags(), eta_formula=eta_formula)
    assert np.allclose(q["eta_avg"], 0.0, atol=1e-10)


def test_eta_formulas_differ_but_are_finite_for_y_dependent_fields():
    fields, dx, dx2 = _make_fields(y_dep=True)
    kwargs = dict(dx=dx, dx2=dx2, avg_axis=1, flags=_default_flags())
    q_thesis = _mean_field_frame_quantities(fields, 0, NoFilter(), eta_formula="thesis", **kwargs)
    q_lhs = _mean_field_frame_quantities(fields, 0, NoFilter(), eta_formula="lhs", **kwargs)
    assert np.all(np.isfinite(q_thesis["eta_avg"]))
    assert np.all(np.isfinite(q_lhs["eta_avg"]))
    assert not np.allclose(q_thesis["eta_avg"], 0.0, atol=1e-8)


def test_e_vlasov_flag_gating():
    fields, dx, dx2 = _make_fields()
    flags_off = AnomalousResistivityConfig(
        species="electrons",
        include_convection=False,
        include_pressure=False,
        include_magnetic_force=False,
    )
    q = _mean_field_frame_quantities(fields, 0, NoFilter(), dx, dx2, avg_axis=1, flags=flags_off, compute_eta=False)
    assert np.allclose(q["e_vlasov_avg"], 0.0)


def test_smoothing_actually_applied_in_frame():
    fields, dx, dx2 = _make_fields()
    filt = GaussianFilter(sigma=3.0)
    q_raw = _mean_field_frame_quantities(fields, 0, NoFilter(), dx, dx2, avg_axis=1, flags=_default_flags())
    q_smooth = _mean_field_frame_quantities(fields, 0, filt, dx, dx2, avg_axis=1, flags=_default_flags())
    assert not np.allclose(q_raw["n_avg"], q_smooth["n_avg"])


# ----------------------------------------------------------------------
# vnT frame pipeline
# ----------------------------------------------------------------------


def test_vnT_frame_labels_and_values():
    fields, dx, dx2 = _make_fields()
    q = _vnT_frame_quantities(fields, 0, NoFilter(), dx, avg_axis=1)

    labels = vnT_feature_labels()
    assert set(q) == set(labels)
    assert len(labels) == len(VNT_BASE_QUANTITIES) * (1 + len(VNT_DERIV_ORDERS))

    n_2d = fields["n"][0]
    nT11_2d = fields["n"][0] * fields["T11"][0]
    assert np.allclose(q["n_avg"], n_2d.mean(axis=1))
    assert np.allclose(q["nT11_avg"], nT11_2d.mean(axis=1))
    # Derivatives are taken on the 2-D data (x = axis 0), then averaged
    assert np.allclose(q["d_n_dx1_avg"], fd_derivative(n_2d, dx, axis=0).mean(axis=1))
    assert np.allclose(q["d4_n_dx1_avg"], fd_derivative(n_2d, dx, axis=0, order=4).mean(axis=1))
    assert np.allclose(q["d3_nT11_dx1_avg"], fd_derivative(nT11_2d, dx, axis=0, order=3).mean(axis=1))


# ----------------------------------------------------------------------
# End-to-end DatabaseCreator on a mock simulation
# ----------------------------------------------------------------------


def test_create_database_all_end_to_end(tmp_path):
    fields, dx, dx2 = _make_fields()
    sim = MockSimulation(fields, (dx, dx2))
    cfg = DatabaseBuildConfig(max_workers=2, flush_every=2)
    db = DatabaseCreator(sim, "electrons", str(tmp_path), build_config=cfg)
    db.set_limits(0, NT)
    db.create_database(database="all")

    input_t = np.load(tmp_path / "input_tensor.npy")
    eta_t = np.load(tmp_path / "eta_tensor.npy")
    vlasov_t = np.load(tmp_path / "e_vlasov_tensor.npy")
    vnT_t = np.load(tmp_path / "vnT_tensor.npy")

    assert input_t.shape == (NT, len(INPUT_FEATURE_LABELS), NX)
    assert eta_t.shape == (NT, len(OUTPUT_LABELS), NX)
    assert vlasov_t.shape == (NT, 1, NX)
    assert vnT_t.shape == (NT, len(vnT_feature_labels()), NX)

    # Values match a direct frame computation
    q = _mean_field_frame_quantities(fields, 1, NoFilter(), dx, dx2, avg_axis=1, flags=_default_flags())
    assert np.allclose(input_t[1], _stack_rows(q, INPUT_FEATURE_LABELS).astype(np.float32), rtol=1e-5)
    assert np.allclose(eta_t[1, 0], q["eta_avg"].astype(np.float32), rtol=1e-5, atol=1e-7)

    # Progress checkpoint removed after a clean finish
    assert not (tmp_path / "input_tensor.npy.progress.npy").exists()


def test_create_database_with_savgol_filter(tmp_path):
    fields, dx, dx2 = _make_fields()
    sim = MockSimulation(fields, (dx, dx2))
    cfg = DatabaseBuildConfig(filters=(SavitzkyGolayFilter(window_length=7, polyorder=4),), eta_formula="lhs")
    db = DatabaseCreator(sim, "electrons", str(tmp_path), build_config=cfg)
    db.set_limits(0, 2)
    db.create_database(database="both")

    input_t = np.load(tmp_path / "input_tensor.npy")
    assert input_t.shape == (2, len(INPUT_FEATURE_LABELS), NX)
    assert np.all(np.isfinite(input_t))


def test_vnT_with_low_savgol_polyorder_raises():
    # The pipeline computes 4th derivatives in a single pass, so a
    # Savitzky-Golay filter needs polyorder >= 4 — no silent chaining.
    fields, dx, dx2 = _make_fields()
    with pytest.raises(ValueError, match="polyorder"):
        _vnT_frame_quantities(fields, 0, SavitzkyGolayFilter(window_length=7, polyorder=2), dx, avg_axis=1)


def test_frame_derivatives_are_single_pass():
    # The d2/d4 features must be the filter's native order-th derivative of
    # the once-smoothed field — not chained first derivatives.
    fields, dx, dx2 = _make_fields()
    filt = GaussianFilter(sigma=2.0)
    n_s = filt.smooth(fields["n"][0], periodic=(False, True))

    q = _mean_field_frame_quantities(fields, 0, filt, dx, dx2, avg_axis=1, flags=_default_flags())
    assert np.allclose(q["d2_n_dx1_avg"], filt.derivative(n_s, dx, axis=0, order=2).mean(axis=1))
    chained = filt.derivative(filt.derivative(n_s, dx, axis=0, order=1), dx, axis=0, order=1)
    assert not np.allclose(q["d2_n_dx1_avg"], chained.mean(axis=1))

    q_vnT = _vnT_frame_quantities(fields, 0, filt, dx, avg_axis=1)
    assert np.allclose(q_vnT["d4_n_dx1_avg"], filt.derivative(n_s, dx, axis=0, order=4).mean(axis=1))


def test_time_derivative_flag_not_supported(tmp_path):
    fields, dx, dx2 = _make_fields()
    sim = MockSimulation(fields, (dx, dx2))
    cfg = DatabaseBuildConfig(ar_config=AnomalousResistivityConfig(species="electrons", include_time_derivative=True))
    db = DatabaseCreator(sim, "electrons", str(tmp_path), build_config=cfg)
    db.set_limits(0, 2)
    with pytest.raises(NotImplementedError):
        db.create_database(database="input")


def test_feature_label_properties():
    db = DatabaseCreator(None, "electrons", "/tmp/unused")
    assert db.feature_labels == INPUT_FEATURE_LABELS
    assert db.output_labels == OUTPUT_LABELS
    assert db.vnT_labels == vnT_feature_labels()


# ----------------------------------------------------------------------
# Lorentz pipeline
# ----------------------------------------------------------------------


def test_boost_frame_beta_zero_reduces_to_unboosted_averages():
    fields, dx, dx2 = _make_fields()
    frame = _boost_combined_frame(fields, 0, beta=0.0, dx=dx, dx2=dx2, avg_axis=1, filt=NoFilter())

    assert frame.shape == (len(LORENTZ_FEATURE_LABELS) + 1, NX)
    assert np.all(np.isfinite(frame))
    # At beta=0 the field transforms are identities
    assert np.allclose(frame[0], fields["n"][0].mean(axis=1))  # n_avg
    assert np.allclose(frame[1], fields["b2"][0].mean(axis=1))  # b2_avg
    assert np.allclose(frame[3], fields["vfl1"][0].mean(axis=1))  # vfl1_avg
    # dvfl1_dx1_avg: derivative on the 2-D field (x = axis 0), then averaged
    assert np.allclose(frame[8], fd_derivative(fields["vfl1"][0], dx, axis=0).mean(axis=1))


def test_boost_frame_with_filter_runs():
    fields, dx, dx2 = _make_fields()
    filt = SavitzkyGolayFilter(window_length=7, polyorder=3)
    frame = _boost_combined_frame(fields, 0, beta=0.5, dx=dx, dx2=dx2, avg_axis=1, filt=filt)
    assert frame.shape == (len(LORENTZ_FEATURE_LABELS) + 1, NX)
    assert np.all(np.isfinite(frame))


def test_lorentz_create_database_end_to_end(tmp_path):
    fields, dx, dx2 = _make_fields()
    sim = MockSimulation(fields, (dx, dx2))
    cfg = LorentzDatabaseBuildConfig(seed=42, filters=(GaussianFilter(sigma=1.0),))
    db = LorentzDatabaseCreator(sim, "electrons", str(tmp_path), build_config=cfg)
    db.set_limits(0, NT)
    db.create_database(database="both")

    input_t = np.load(tmp_path / "lorentz_tensor.npy")
    output_t = np.load(tmp_path / "lorentz_output.npy")
    betas = np.load(tmp_path / "boost_velocities.npy")

    assert input_t.shape == (NT, len(LORENTZ_FEATURE_LABELS), NX)
    assert output_t.shape == (NT, 1, NX)
    assert betas.shape == (NT,)
    assert np.all(np.isfinite(input_t))
    assert np.all(np.isfinite(output_t))

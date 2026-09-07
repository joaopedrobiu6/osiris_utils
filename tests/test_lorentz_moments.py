"""Ground-truth checks of the boosted moments and the momentum equation.

Everything else in the suite compares the pipeline against *itself* on made-up
profiles.  These two tests compare it against physics:

``test_boosted_moments_match_quadrature``
    Takes an actual distribution function on a 3-D proper-velocity grid,
    computes every lab moment with the OSIRIS deposit definitions and every
    boosted moment with the note's Eq. 13 Jacobian, and checks ``_boost_fields``
    reproduces the second set.  Made-up moment profiles cannot catch an error
    here, because independently invented n, vfl, ufl, P11, P00 do not come from
    any f and so satisfy no transformation law.

``test_e_vlasov_is_zero_and_boost_invariant_on_a_vlasov_solution``
    ``f(x,t,u) = g(u) h(x - v_x t)`` solves the Vlasov equation exactly with
    E = B = 0, so the momentum-equation residual -- which is what e_vlasov
    reconstructs as E_x -- must vanish, in the lab frame and in every boosted
    frame (E'_x = E_x, note Eq. 29).  This is what pins ``<u_1>`` rather than
    ``<v_1>`` as the advected quantity: the non-relativistic form leaves a
    residual three orders of magnitude larger and is not boost-invariant.

Both are quadrature-limited, hence the 1e-5 tolerances on 1e-16-exact algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from osiris_utils.ar import AnomalousResistivityConfig
from osiris_utils.database.lorentz_database import _boost_fields, _boost_frame_quantities
from osiris_utils.filters import NoFilter

BETAS = [0.0, 0.3, 0.6, 0.8]


# ---------------------------------------------------------------------------
# 1. Moment transforms vs. direct quadrature over a distribution function
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moments():
    """Lab and boosted moments of one drifting distribution, by quadrature.

    ``g`` is a Gaussian in proper velocity -- not a Juttner, which does not
    matter: the transformation laws are exact for *any* f, so any f tests them.
    """
    n_u, u_max = 121, 2.5
    ax = np.linspace(-u_max, u_max, n_u)
    du = ax[1] - ax[0]
    u1, u2, u3 = np.meshgrid(ax, ax, ax, indexing="ij")
    gp = np.sqrt(1.0 + u1**2 + u2**2 + u3**2)  # gamma_p
    v1, v2, v3 = u1 / gp, u2 / gp, u3 / gp
    f = np.exp(-((u1 - 0.45) ** 2 + (u2 - 0.20) ** 2 + u3**2) / (2 * 0.32**2))

    def integrate(q):
        return float((f * q).sum() * du**3)

    n = integrate(1.0)
    # OSIRIS deposits (source/spec/os-spec-udist.f03): vfl/ufl are normalised by
    # the charge density, the P moments are not -- so P11 = n<v_1 u_1>,
    # P12 = n<v_1 u_2>, P00 = n<gamma_p>.
    lab = {
        "n": n,
        "vfl1": integrate(v1) / n,
        "vfl2": integrate(v2) / n,
        "vfl3": integrate(v3) / n,
        "ufl1": integrate(u1) / n,
        "ufl2": integrate(u2) / n,
        "P11": integrate(v1 * u1),
        "P12": integrate(v1 * u2),
        "P00": integrate(gp),
    }

    exact = {}
    for beta in BETAS:
        g = 1.0 / np.sqrt(1.0 - beta**2)
        jac = g * (1.0 - beta * v1)  # note Eq. 13
        u1p, u2p = g * (u1 - beta * gp), u2
        gpp = g * (gp - beta * u1)  # = sqrt(1 + u'^2)
        v1p, v2p = u1p / gpp, u2p / gpp

        def integrate_b(q, jac=jac):
            return float((f * q * jac).sum() * du**3)

        n_p = integrate_b(1.0)
        v1_p, v2_p = integrate_b(v1p) / n_p, integrate_b(v2p) / n_p
        u1_p = integrate_b(u1p) / n_p
        # Pi'_11 and Pi'_12 as the momentum equation needs them:
        # Pi'_1j = n'<v'_j u'_1> - n'<v'_j><u'_1>.
        exact[beta] = {
            "n": n_p,
            "vfl1": v1_p,
            "vfl2": v2_p,
            "ufl1": u1_p,
            "T11": (integrate_b(v1p * u1p) - n_p * v1_p * u1_p) / n_p,
            "T12": (integrate_b(v2p * u1p) - n_p * v2_p * u1_p) / n_p,
        }
    return lab, exact


@pytest.mark.parametrize("beta", BETAS)
def test_boosted_moments_match_quadrature(moments, beta):
    """Every quantity _boost_fields returns, against the integral that defines it."""
    lab, exact = moments
    g = 1.0 / np.sqrt(1.0 - beta**2)
    fields = {k: np.array([[v]], dtype=float) for k, v in {**lab, "e2": 0.0, "e3": 0.0, "b2": 0.0, "b3": 0.0}.items()}
    got = _boost_fields(fields, beta, g)

    for name, want in exact[beta].items():
        assert got[name][0, 0] == pytest.approx(want, rel=1e-5), f"{name} at beta={beta}"


def test_transverse_velocity_is_not_invariant(moments):
    """Note Eqs. 17-18 say <v'_y> = <v_y>; its own Eq. 13 says otherwise.

    ``v'_y * det(J) = u_y / gamma_p = v_y`` pointwise, so
    ``n'<v'_y> = n<v_y>`` and the transverse velocity carries the density ratio.
    Leaving it unchanged is wrong by ~6% at beta = 0.3 and ~20% at beta = 0.8,
    and it breaks the boost invariance of e_vlasov through ``-v_y B'_z``.
    """
    lab, exact = moments
    for beta, tol in ((0.3, 0.05), (0.8, 0.15)):
        naive = lab["vfl2"]  # what Eqs. 17-18 claim
        assert abs(naive - exact[beta]["vfl2"]) / abs(exact[beta]["vfl2"]) > tol
        # and the density-ratio form is exact
        g = 1.0 / np.sqrt(1.0 - beta**2)
        assert lab["n"] * lab["vfl2"] / exact[beta]["n"] == pytest.approx(exact[beta]["vfl2"], rel=1e-6)
        assert lab["vfl2"] / (g * (1.0 - beta * lab["vfl1"])) == pytest.approx(exact[beta]["vfl2"], rel=1e-6)


# ---------------------------------------------------------------------------
# 2. e_vlasov on an exact Vlasov solution
# ---------------------------------------------------------------------------

NX, NY, LX = 48, 4, 6.0
DT, T0 = 0.02, 1.0


@pytest.fixture(scope="module")
def free_streaming():
    """Moments of ``f = g(u) h(x - v_x t)`` at three times, as a raw-diagnostic dict.

    Uniform in y, so every d/dy term vanishes identically and the test isolates
    the longitudinal momentum equation.
    """
    n_u, u_max = 55, 2.5
    x = np.linspace(0.0, LX, NX, endpoint=False)
    dx = x[1] - x[0]
    ax = np.linspace(-u_max, u_max, n_u)
    du = ax[1] - ax[0]
    u1, u2, u3 = np.meshgrid(ax, ax, ax, indexing="ij")
    gp = np.sqrt(1.0 + u1**2 + u2**2 + u3**2)
    v1 = u1 / gp
    g = np.exp(-((u1 - 0.35) ** 2 + (u2 - 0.15) ** 2 + u3**2) / (2 * 0.30**2))

    def level(t):
        # h(x - v_x t): the exact free-streaming solution of h(x) at t = 0.
        w = g * (1.0 + 0.30 * np.sin(2 * np.pi * (x[:, None, None, None] - v1[None] * t) / LX))

        def integrate(q):
            return (w * q).sum(axis=(1, 2, 3)) * du**3

        n = integrate(1.0)
        return {
            "n": n,
            "vfl1": integrate(v1) / n,
            "vfl2": integrate(u2 / gp) / n,
            "vfl3": integrate(u3 / gp) / n,
            "ufl1": integrate(u1) / n,
            "ufl2": integrate(u2) / n,
            "P11": integrate(v1 * u1),
            "P12": integrate(v1 * u2),
            "P00": integrate(gp),
        }

    levels = [level(T0 - DT), level(T0), level(T0 + DT)]
    raw = {k: np.repeat(np.stack([lv[k] for lv in levels])[:, :, None], NY, axis=2) for k in levels[0]}
    raw |= {k: np.zeros((3, NX, NY)) for k in ("e2", "e3", "b2", "b3")}
    return raw, dx


@pytest.mark.parametrize("beta", [0.0, 0.3, 0.6])
def test_e_vlasov_is_zero_and_boost_invariant_on_a_vlasov_solution(free_streaming, beta):
    """E = B = 0, so the reconstructed E'_x must be 0 in every frame.

    This is the test that distinguishes the relativistic momentum equation from
    the non-relativistic one.  With ``<u_1>`` advected the residual is ~1e-6
    (quadrature-limited) and flat in beta; with ``<v_1>`` it is ~5e-3 and varies
    with beta, because E'_x = E_x cannot hold for the wrong equation.
    """
    raw, dx = free_streaming
    flags = AnomalousResistivityConfig(
        include_time_derivative=True,
        include_convection=True,
        include_pressure=True,
        include_magnetic_force=True,
    )
    q = _boost_frame_quantities(
        raw,
        {k: (0, 1, 2) for k in raw},
        2 * DT,
        beta,
        NoFilter(),
        dx,
        LX / NY,
        avg_axis=1,
        flags=flags,
        compute_e_vlasov=True,
        compute_eta=False,
        rqm=-1.0,
    )
    interior = slice(6, NX - 6)  # skip the one-sided finite-difference edges
    assert np.abs(q["e_vlasov_avg"][interior]).max() < 1e-4

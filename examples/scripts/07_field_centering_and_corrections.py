r"""Grid- and moment-level corrections: Yee centering, pressure, heat flux.

Three post-processors that fix a systematic offset in what OSIRIS writes.

**Field centering.**  OSIRIS solves Maxwell's equations on a Yee mesh, so each
field component is stored at a different point of the cell — ``E_i`` on the
edge, ``B_i`` on the face.  Multiplying ``e1 * vfl1`` therefore multiplies
quantities that do not live at the same place.
:class:`~osiris_utils.FieldCentering_Diagnostic` averages the two neighbouring
values along each staggered axis to bring every component to the cell centre.
``periodic=True`` wraps at the boundary; ``periodic=False`` repeats the edge
value, which is what an open boundary can support.

**Pressure correction.**  OSIRIS's ``P_jk`` is the second moment in the lab
frame.  Subtracting the bulk flow gives the pressure in the fluid rest frame,

.. math:: \Pi_{jk} = P_{jk} - n\,u_j\,v_k,

with :math:`u_j` the proper velocity (``ufl``) and :math:`v_k` the fluid
velocity (``vfl``) — different quantities, so a run that only dumped ``vfl``
gets a warning, not a silent substitution.

**Heat-flux correction.**  Same idea one moment up: ``Q_ijk`` minus the flux
carried by the bulk motion and by the pressure transported at the bulk speed.

Run::

    python examples/scripts/07_field_centering_and_corrections.py [--sim DECK] [--plot]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]

    # ==================================================================
    section("1. Field centering")
    # ==================================================================
    centered_sim = ou.FieldCentering_Simulation(sim, periodic=True)
    b3_c = centered_sim["b3"]

    show("centered b3 frame", b3_c[3].shape)
    show("shape is unchanged", b3_c[3].shape == sim["b3"][3].shape)
    show("max shift", float(np.abs(b3_c[3] - sim["b3"][3]).max()))

    # Which axes a component is staggered on follows from the Yee mesh, so the
    # wrapper only accepts field names.
    try:
        centered_sim["n"]
    except ValueError as e:
        show("only fields can be centered", f"ValueError: {str(e)[:60]}...")

    # Open boundaries: the edge value is repeated rather than wrapped.
    open_centered = ou.FieldCentering_Simulation(sim, periodic=False)
    show("periodic=False edge differs", not np.allclose(open_centered["b3"][3][0], b3_c[3][0]))

    # Direct construction, and the whole series at once.
    e1_c = ou.FieldCentering_Diagnostic(sim["e1"], periodic=True)
    e1_c.load_all()
    show("load_all", e1_c.data.shape)
    e1_c.unload()

    centered_sim.delete("b3")
    centered_sim.delete_all()

    # ==================================================================
    section("2. Pressure correction")
    # ==================================================================
    pc = ou.PressureCorrection_Simulation(sim)
    P12 = sim[species]["P12"]
    P12_c = pc[species]["P12"]

    show("corrected name", P12_c.name)
    show("frame", P12_c[3].shape)

    # Verify the formula on one frame: Pi_12 = P_12 - n u_1 v_2
    n = sim[species]["n"][3]
    u1 = sim[species]["ufl1"][3]
    v2 = sim[species]["vfl2"][3]
    show("== P12 - n*ufl1*vfl2", np.allclose(P12_c[3], P12[3] - n * u1 * v2))

    show("every component", [pc[species][k][3].shape for k in ("P11", "P22", "P33")])

    # Pressure is species-dependent, so it has to be reached through a species.
    try:
        pc["P12"]
    except (KeyError, ValueError) as e:
        show("sim-level access", f"{type(e).__name__}: {str(e)[:60]}...")

    P12_c.load_all()
    show("load_all", P12_c.data.shape)
    P12_c.unload()

    # ==================================================================
    section("3. Heat-flux correction")
    # ==================================================================
    hc = ou.HeatfluxCorrection_Simulation(sim)
    for key in ("Q111", "Q112", "Q223"):
        try:
            show(f"{key} corrected", hc[species][key][3].shape)
        except (FileNotFoundError, ValueError, KeyError) as e:
            show(f"{key} unavailable", type(e).__name__)

    # The formula, for Q_ijk:
    #   q_ijk = Q_ijk - (v_i P_jk + v_j P_ki + v_k P_ij) + 2 v_i v_j v_k n
    # OSIRIS only dumps the upper triangle of the symmetric pressure tensor, so
    # a mixed component orders each index pair (P12, never P21).
    v1, v2 = sim[species]["vfl1"][3], sim[species]["vfl2"][3]
    p11, p12 = sim[species]["P11"][3], sim[species]["P12"][3]
    manual = sim[species]["Q112"][3] - (v1 * p12 + v1 * p12 + v2 * p11) + 2 * v1 * v1 * v2 * n
    show("Q112 matches the formula", np.allclose(hc[species]["Q112"][3], manual))

    # ==================================================================
    section("4. Composing corrections with everything else")
    # ==================================================================
    # Every corrected quantity is a Diagnostic, so it composes with arithmetic,
    # derivatives, MFT and filters exactly like a raw one.
    pi12 = pc[species]["P12"]
    d_pi12 = ou.Derivative_Diagnostic(pi12, "x1", order=4)
    show("d(Pi_12)/dx1", d_pi12[3].shape)
    if pi12.dim >= 2:
        show("<Pi_12>_x2", ou.MFT_Diagnostic(pi12, mft_axis=2)["avg"][3].shape)
    show("centered E1 * vfl1", (ou.FieldCentering_Diagnostic(sim["e1"]) * sim[species]["vfl1"])[3].shape)

    # ==================================================================
    section("5. A plot")
    # ==================================================================
    import matplotlib.pyplot as plt

    x = P12.x[0] if P12.dim > 1 else P12.x
    raw = P12[3].mean(axis=1) if P12.dim > 1 else P12[3]
    corr = P12_c[3].mean(axis=1) if P12.dim > 1 else P12_c[3]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, raw, label=r"$P_{12}$ (lab frame)")
    ax.plot(x, corr, label=r"$\Pi_{12} = P_{12} - n u_1 v_2$")
    ax.set_xlabel(P12.axis[0]["plot_label"])
    ax.set_ylabel("pressure")
    ax.legend()
    savefig(fig, "07_pressure_correction.png", args)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

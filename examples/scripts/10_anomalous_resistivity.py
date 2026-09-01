r"""Anomalous resistivity: the mean-field residual of the momentum equation.

Solving the species momentum equation for the longitudinal electric field,

.. math::
    \frac{m}{q}\left[\partial_t v_1 + v_1\partial_1 v_1 + v_2\partial_2 v_1
        + \frac{\partial_1(nT_{11}) + \partial_2(nT_{12})}{n}\right]
    = E_1 + (v\times B)_1 ,

defines ``e_vlasov`` — the field the fluid equation *requires*.  Every term
carries ``rqm = m/q`` (-1 for electrons, +32 for the shock-deck ions) except the
magnetic one, because E and v x B are divided by q together.  ``rqm`` is read
from the input deck, so an ion result is an ion result and not an electron
result computed on ion data.

Splitting each quantity into a transverse mean and a fluctuation and averaging
leaves correlations of fluctuations that the mean-field equation cannot
produce.  Their sum is the anomalous resistivity ``eta``: the effective drag the
turbulence exerts on the mean flow.

``AnomalousResistivity`` gives you all of it as lazy diagnostics — `LHS` and
its individual contributions, `eta` (the 7-term thesis pressure decomposition),
`eta_new` (the simplified 4-term form), and every cross-term separately,
**already multiplied by its coefficient**, so a stacked plot of the terms adds
up to the total with no further sign or ``rqm`` bookkeeping.

Run::

    python examples/scripts/10_anomalous_resistivity.py [--sim DECK] [--plot]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, savefig, section, show

import osiris_utils as ou


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]
    t = 3

    if sim["e1"].dim < 2:
        print("This example needs a 2-D run (mean-field theory averages over x2).")
        return

    # ==================================================================
    section("1. Building it")
    # ==================================================================
    # The defaults: convection, pressure and magnetic force on, no time
    # derivative (it needs burst dumps — see example 11), transverse advection
    # off, averaging over x2, no spatial filtering.
    ar = ou.AnomalousResistivity(sim, species)

    show("species / rqm", (ar.species, sim[species].species.rqm))
    show("e_vlasov key on the Simulation", ar.e_vlasov_key)
    show("terms available", len(ar.available_terms))
    # ar.x / ar.dx are the *longitudinal* axis: every mean-field quantity is
    # a function of x1 alone once the transverse average has been taken.
    show("x (longitudinal axis)", ar.x)
    show("dx (longitudinal spacing)", ar.dx)
    show("filter (none by default)", repr(ar.filter))
    # Without filters this is the Simulation itself; with them, a filtered view.
    show("filtered_simulation", type(ar.filtered_simulation).__name__)

    # ==================================================================
    section("2. The Vlasov field and the LHS")
    # ==================================================================
    e_vlasov = sim[ar.e_vlasov_key]  # full 2-D field, lazily evaluated
    show("e_vlasov frame", e_vlasov[t].shape)
    show("<e_vlasov>", ar["e_vlasov_avg"][t].shape)

    # LHS is exactly the sum of its lhs_* contributions, each already weighted.
    lhs_terms = {k: v for k, v in ar.terms_dict.items() if k.startswith("lhs_")}
    total = sum(v[t] for v in lhs_terms.values())
    show("lhs_* terms", list(lhs_terms))
    show("sum(lhs_*) == LHS", np.allclose(total, ar["LHS"][t]))

    # ==================================================================
    section("3. eta and eta_new")
    # ==================================================================
    show("eta frame", ar["eta"][t].shape)
    show("eta_new frame", ar["eta_new"][t].shape)

    # The fluctuation cross-terms, grouped by physical origin.
    groups = {"convection": "conv_", "magnetic": "mag_", "pressure (thesis)": "press_dnT", "pressure (new)": "press_new"}
    for label, prefix in groups.items():
        names = [k for k in ar.available_terms if k.startswith(prefix)]
        show(label, names)

    # Every term is pre-multiplied by its coefficient, so eta is their plain sum.
    thesis_terms = [k for k in ar.available_terms if k.startswith(("conv_", "mag_", "press_")) and not k.startswith("press_new")]
    show("sum of thesis terms == eta", np.allclose(sum(ar[k][t] for k in thesis_terms), ar["eta"][t]))
    show("some coefficients", dict(list(ar.term_coefficients.items())[:6]))

    # The unweighted MFT diagnostics, if you want the raw correlation instead.
    # These are MFT containers: index them with "avg" or "delta" first.  Each
    # weighted term above is coefficient * <that correlation>.
    show("mft_terms (unweighted)", list(ar.mft_terms)[:4])
    raw_corr = ar.mft_terms["conv_v1_dv1dx1"]["avg"]
    show(
        "weighted == coeff * <correlation>",
        np.allclose(
            ar["conv_v1_dv1dx1"][t],
            ar.term_coefficients["conv_v1_dv1dx1"] * raw_corr[t],
        ),
    )

    # ==================================================================
    section("4. Plain averages, for context")
    # ==================================================================
    for key in ("n_avg", "vfl1_avg", "b3_avg", "dnT11_dx1_avg"):
        show(key, ar[key][t].shape)
    try:
        ar["not_a_term"]
    except KeyError as e:
        show("unknown term", f"KeyError: {str(e)[:48]}...")

    # ==================================================================
    section("5. Turning terms on and off")
    # ==================================================================
    # Each flag removes the term from e_vlasov *and* its cross-terms from eta,
    # so the two stay consistent.
    configs = {
        "defaults": ou.AnomalousResistivityConfig(species=species),
        "+ transverse advection": ou.AnomalousResistivityConfig(species=species, include_transverse_advection=True),
        "no magnetic force": ou.AnomalousResistivityConfig(species=species, include_magnetic_force=False),
        "no pressure": ou.AnomalousResistivityConfig(species=species, include_pressure=False),
        "convection only": ou.AnomalousResistivityConfig(species=species, include_pressure=False, include_magnetic_force=False),
    }
    # Note the warnings on stderr: each new configuration finds an e_vlasov
    # already on this Simulation and stores its own under a suffixed key rather
    # than reusing one built from different terms.  In real work, use one
    # Simulation per configuration.
    for label, cfg in configs.items():
        a = ou.AnomalousResistivity(sim, species, config=cfg)
        show(label, f"{len(a.available_terms):2d} terms   <eta> = {float(a['eta'][t].mean()):+.4e}")

    # Each configuration gets its own e_vlasov on the Simulation, so two
    # AnomalousResistivity objects never share one built for other flags.
    a2 = ou.AnomalousResistivity(sim, species, config=configs["no pressure"])
    show("distinct e_vlasov keys", (ar.e_vlasov_key, a2.e_vlasov_key))

    # ==================================================================
    section("6. Other species, and overriding rqm")
    # ==================================================================
    if len(sim.species) > 1:
        ion = sim.species[1]
        ar_ion = ou.AnomalousResistivity(sim, ion)
        show(f"{ion}: rqm from the deck", sim[ion].species.rqm)
        show(f"{ion}: <eta>", float(ar_ion["eta"][t].mean()))
        show("electron <eta> for comparison", float(ar["eta"][t].mean()))

    # For a deck that cannot be parsed, pass rqm explicitly.
    ar_manual = ou.AnomalousResistivity(sim, species, rqm=-1.0)
    show("rqm=-1.0 override", float(ar_manual["eta"][t].mean()))

    # ==================================================================
    section("7. Filtering, per frame")
    # ==================================================================
    # config.filters runs the database's spatial filters through the lazy
    # pipeline: each raw frame is smoothed on its way off disk, and every
    # spatial derivative uses that filter's own single-pass kernel.  The result
    # reproduces the tensor column the database would have written for this
    # timestep — with nothing precomputed.
    filt = ou.SavitzkyGolayFilter(window_length=9, polyorder=4)
    ar_filtered = ou.AnomalousResistivity(sim, species, config=ou.AnomalousResistivityConfig(species=species, filters=filt))
    show("filter in use", repr(ar_filtered.filter))
    show("frames come from a filtered view", type(ar_filtered.simulation).__name__)
    show("<eta> unfiltered vs filtered", (float(ar["eta"][t].mean()), float(ar_filtered["eta"][t].mean())))

    # ==================================================================
    section("8. The convenience function")
    # ==================================================================
    # Just the field, when the decomposition is not what you are after.
    ev = ou.vlasov_electric_field(sim, species)
    show("vlasov_electric_field", ev[t].shape)

    # ==================================================================
    section("9. A plot: what eta is made of")
    # ==================================================================
    import matplotlib.pyplot as plt

    x = ar.x
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5))

    ax1.plot(x, ar["LHS"][t].ravel(), label="LHS")
    ax1.plot(x, ar["e_vlasov_avg"][t].ravel(), "--", label=r"$\langle e_{vlasov}\rangle$")
    ax1.set_ylabel("field")
    ax1.legend(fontsize=8)

    for key in thesis_terms:
        ax2.plot(x, ar[key][t].ravel(), lw=0.9, label=key)
    ax2.plot(x, ar["eta"][t].ravel(), "k", lw=1.8, label=r"$\eta$ (their sum)")
    ax2.set_xlabel(r"$x_1\ [c/\omega_p]$")
    ax2.set_ylabel(r"$\eta$ contributions")
    ax2.legend(fontsize=6, ncol=2)
    savefig(fig, "10_anomalous_resistivity.png", args)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

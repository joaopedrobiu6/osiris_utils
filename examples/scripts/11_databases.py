r"""Building the training tensors: DatabaseCreator, bursts, Lorentz augmentation.

A "database" here is a labelled ``(T, F, X)`` NumPy tensor — T dumps, F
features, X longitudinal cells — that a model can be trained on.  Each frame is
built the same way the physics demands:

    raw 2-D frames  ->  spatial filter  ->  all derivatives and e_vlasov in 2-D
                    ->  transverse average  ->  stack the labelled rows

Derivatives are taken **before** the average, because the average of a
derivative of a fluctuating field is not the derivative of the average.

Tensors available
-----------------
``"input"``     mean-field features (``INPUT_FEATURE_LABELS``) — 46 rows.
``"output"``    ``eta``, from ``eta_formula="thesis"`` (default) or ``"lhs"``.
``"e_vlasov"``  the mean-field Vlasov field.
``"vnT"``       v, n, T and their x1-derivatives up to 4th order.
``"InOut"``     input + output in a single pass.  ``"all"`` adds the other two.

Everything requested is built in **one** pass — each field is read and filtered
exactly once per timestep — and streamed to a memory-mapped ``.npy``, so the
full tensor never sits in RAM and an interrupted job resumes.

Run::

    python examples/scripts/11_databases.py [--sim DECK]
"""

from __future__ import annotations

import numpy as np
from _common import load_simulation, parse_args, section, show

import osiris_utils as ou
from osiris_utils.database import describe_axes, input_feature_labels
from osiris_utils.database.database import vnT_feature_labels


def main() -> None:
    args = parse_args(__doc__)
    sim = load_simulation(args)
    species = sim.species[0]
    out = args.outdir / "db"
    out.mkdir(parents=True, exist_ok=True)

    if sim["e1"].dim < 2:
        print("This example needs a 2-D run.")
        return

    n_dumps = len(sim["e1"])

    # ==================================================================
    section("1. The simplest build")
    # ==================================================================
    db = ou.DatabaseCreator(sim, species, out / "basic")
    db.set_limits(0, n_dumps)  # [initial_iter, final_iter) in dump indices
    show("T (dumps) / X (cells)", (db.T, db.X))
    # x / dx are the longitudinal axis the tensor columns sit on.
    show("x (longitudinal axis)", db.x)
    show("dx", db.dx)

    db.create_database(database="InOut")
    inp = np.load(out / "basic" / "input_tensor.npy")
    eta = np.load(out / "basic" / "eta_tensor.npy")
    show("input tensor", inp.shape)
    show("eta tensor", eta.shape)
    show("rows == labels", inp.shape[1] == len(db.feature_labels))
    show("first labels", db.feature_labels[:6])
    show("output labels", db.output_labels)

    # ==================================================================
    section("2. Every tensor type")
    # ==================================================================
    for kind in ("input", "output", "e_vlasov", "vnT"):
        creator = ou.DatabaseCreator(sim, species, out / kind)
        creator.set_limits(0, n_dumps)
        creator.create_database(database=kind)
        files = sorted(p.name for p in (out / kind).glob("*.npy"))
        show(kind, f"{files} {np.load(out / kind / files[0]).shape}")

    show("vnT rows", vnT_feature_labels()[:6])
    show("input rows for the default flags", len(input_feature_labels()))

    # "all" writes every tensor from one pass over the frames.
    every = ou.DatabaseCreator(sim, species, out / "all")
    every.set_limits(0, n_dumps)
    every.create_database(
        database="all",
        name_input="X",
        name_output="y",
        name_vlasov="ev",
        name_vnT="vnT",
    )
    show("custom file stems", sorted(p.name for p in (out / "all").glob("*.npy")))

    # ==================================================================
    section("3. Build options")
    # ==================================================================
    cfg = ou.DatabaseBuildConfig(
        dtype=np.float32,  # storage dtype (float64 doubles the file)
        max_workers=4,  # frame-building threads
        mft_axis=2,  # average over x2 (OSIRIS 1-indexed)
        eta_formula="thesis",  # or "lhs": <e_vlasov> plus mean-field corrections
        validate_output=True,  # NaN/inf -> 0 in the eta / e_vlasov tensors, logged
        flush_every=64,  # checkpoint cadence; 1 for an unstable HPC queue
        resume=False,  # True skips frames a previous run already wrote
        rqm=None,  # None reads m/q from the deck
    )
    tuned = ou.DatabaseCreator(sim, species, out / "tuned", cfg)
    tuned.set_limits(0, n_dumps)
    tuned.create_database(database="output")
    show("thesis eta", np.load(out / "tuned" / "eta_tensor.npy").shape)

    lhs_cfg = ou.DatabaseBuildConfig(eta_formula="lhs")
    lhs = ou.DatabaseCreator(sim, species, out / "lhs", lhs_cfg)
    lhs.set_limits(0, n_dumps)
    lhs.create_database(database="output")
    thesis_eta = np.load(out / "tuned" / "eta_tensor.npy")
    lhs_eta = np.load(out / "lhs" / "eta_tensor.npy")
    show("thesis vs lhs <eta>", (float(thesis_eta.mean()), float(lhs_eta.mean())))

    # A slice of the run, by dump index:
    part = ou.DatabaseCreator(sim, species, out / "slice")
    part.set_limits(2, 5)
    part.create_database(database="input")
    show("set_limits(2, 5)", np.load(out / "slice" / "input_tensor.npy").shape)

    # ==================================================================
    section("4. Physics flags and filtering")
    # ==================================================================
    # ar_config gates which terms enter e_vlasov and eta — the same flags as
    # AnomalousResistivity (example 10).
    flags = ou.AnomalousResistivityConfig(species=species, include_transverse_advection=True)
    with_adv = ou.DatabaseCreator(sim, species, out / "adv", ou.DatabaseBuildConfig(ar_config=flags))
    with_adv.set_limits(0, n_dumps)
    with_adv.create_database(database="output")
    show("with transverse advection", np.load(out / "adv" / "eta_tensor.npy").shape)

    # filters smooth every raw frame before any physics, and supply the
    # derivative scheme for every order.  The pipeline takes 4th derivatives,
    # so a Savitzky-Golay filter needs polyorder >= 4.
    filt_cfg = ou.DatabaseBuildConfig(filters=ou.SavitzkyGolayFilter(window_length=9, polyorder=4))
    filtered = ou.DatabaseCreator(sim, species, out / "filtered", filt_cfg)
    filtered.set_limits(0, n_dumps)
    filtered.create_database(database="InOut")
    show("filtered eta", np.load(out / "filtered" / "eta_tensor.npy").shape)
    show(
        "unfiltered vs filtered <eta>",
        (
            float(np.load(out / "basic" / "eta_tensor.npy").mean()),
            float(np.load(out / "filtered" / "eta_tensor.npy").mean()),
        ),
    )
    # The filter can equally come from ar_config.filters — but not from both
    # with different values, which is an error rather than a silent choice.

    # ==================================================================
    section("5. Burst dumps and the time derivative")
    # ==================================================================
    # d(vfl1)/dt taken across ordinary dumps is a derivative over dt * ndump,
    # not the derivative *at* the dump time, so the term is refused unless the
    # run was written with burst dumps (`if_use_burst_dump`, `burst_dump_range`),
    # which add frames at n +/- 1 around every ordinary dump.
    try:
        bad = ou.DatabaseBuildConfig(ar_config=ou.AnomalousResistivityConfig(species=species, include_time_derivative=True))
        c = ou.DatabaseCreator(sim, species, out / "nope", bad)
        c.set_limits(0, n_dumps)
        c.create_database(database="input")
    except NotImplementedError as e:
        show("d/dt without bursts", f"NotImplementedError: {str(e)[:60]}...")

    if args.sim:
        show("burst demo skipped", "pass a burst-dumped run to try it")
    else:
        burst_sim = load_simulation(args, burst=True)
        vfl1 = burst_sim[species]["vfl1"]
        show("burst iteration axis", vfl1.iterations)
        show("stride (1 = filenames are absolute iterations)", vfl1._iter_stride)

        burst_cfg = ou.DatabaseBuildConfig(
            burst=ou.BurstConfig(
                ndump_fac=1,  # ndump_fac of the diagnostic
                deriv_quantities=("vfl1",),  # must be bursted in the deck
                require_centered=True,  # drop n=0, which has no left neighbour
            ),
            ar_config=ou.AnomalousResistivityConfig(species=species, include_time_derivative=True),
        )
        bdb = ou.DatabaseCreator(burst_sim, species, out / "burst", burst_cfg)
        bdb.set_limits(0, len(burst_sim["e1"]))
        bdb.create_database(database="input")
        show("burst input tensor", np.load(out / "burst" / "input_tensor.npy").shape)
        show("extra row from d/dt", bdb.feature_labels[-1])

        # BurstAxis on its own: it aligns diagnostics on the midpoint (ordinary
        # dump) iterations and hands back the frame pair for a centred
        # in-burst d/dt, so the term lands on the same time and grid points as
        # every other one.
        axis = ou.BurstAxis(
            {"vfl1": vfl1, "b3": burst_sim["b3"]},
            dt=vfl1.dt,
            ndump=vfl1.ndump,
            config=ou.BurstConfig(ndump_fac=1),
        )
        show("midpoints", axis.midpoints)
        show("dump_indices", axis.dump_indices)
        # A sanity check before a long build: which diagnostics were bursted.
        print(describe_axes({"vfl1": vfl1, "b3": burst_sim["b3"]}, ndump=vfl1.ndump, ndump_fac=1))

    # ==================================================================
    section("6. Lorentz-augmented database")
    # ==================================================================
    # A random x-boost is drawn per timestep and every field is transformed
    # before the transverse average, multiplying the training set without
    # running another simulation.  beta is saved, so a resumed job reuses it.
    lcfg = ou.LorentzDatabaseBuildConfig(
        boost_min=0.0,
        boost_max=0.5,  # boost_max must stay < 1 (gamma diverges)
        seed=12345,  # reproducible augmentation
        mft_axis=2,
    )
    ldb = ou.LorentzDatabaseCreator(sim, species, out / "lorentz", lcfg)
    ldb.set_limits(0, n_dumps)
    ldb.create_database(database="both")

    show("input tensor", np.load(out / "lorentz" / "lorentz_tensor.npy").shape)
    show("output tensor", np.load(out / "lorentz" / "lorentz_output.npy").shape)
    show("betas", np.load(out / "lorentz" / "boost_velocities.npy"))
    show("feature labels", ldb.feature_labels[:5])
    show("output labels", ldb.output_labels)

    # ==================================================================
    section("7. Using a tensor")
    # ==================================================================
    X = np.load(out / "basic" / "input_tensor.npy")
    y = np.load(out / "basic" / "eta_tensor.npy")
    labels = db.feature_labels

    show("X / y", (X.shape, y.shape))
    # Flatten (T, F, X) -> (T*X, F) for a per-cell regression:
    X_flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    y_flat = y.transpose(0, 2, 1).reshape(-1)
    show("flattened for training", (X_flat.shape, y_flat.shape))

    # The strongest linear correlations with eta.  On this synthetic run every
    # mean-field quantity jumps at the same front, so they all correlate
    # strongly — on real data this is a quick way to see which features carry
    # signal before training anything.
    corr = [(labels[i], float(np.corrcoef(X_flat[:, i], y_flat)[0, 1])) for i in range(X.shape[1])]
    corr = sorted(corr, key=lambda kv: -abs(kv[1] if np.isfinite(kv[1]) else 0))[:5]
    for name, c in corr:
        show(name, f"corr(eta) = {c:+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

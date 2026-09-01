"""Reading and rewriting OSIRIS input decks with InputDeckIO.

An OSIRIS deck is a list of named namelist sections, and the *same section name
can appear many times* (one ``species`` block per species).  ``InputDeckIO``
keeps that structure: ``get_param`` always returns a **list**, one entry per
matching section, and ``set_param`` writes to all of them unless told otherwise.

This is what makes parameter scans scriptable: parse a template, substitute,
write a new deck per job.

Run::

    python examples/scripts/03_input_decks.py [--sim DECK]
"""

from __future__ import annotations

from _common import deck_path, parse_args, section, show

import osiris_utils as ou
from osiris_utils.decks.decks import deval


def main() -> None:
    args = parse_args(__doc__)
    deck_file = deck_path(args)
    print(f"Input deck      : {deck_file}")

    # ------------------------------------------------------------------
    section("1. Parsing")
    # ------------------------------------------------------------------
    # verbose=True echoes every section and parameter as it is read — the fast
    # way to find the line a malformed deck chokes on.
    deck = ou.InputDeckIO(str(deck_file), verbose=False)

    show("filename", deck.filename)
    show("dim (from nx_p(1:d))", deck.dim)
    show("n_species", deck.n_species)
    show("section names", [name for name, _ in deck.sections])

    # ------------------------------------------------------------------
    section("2. Species")
    # ------------------------------------------------------------------
    # `species` is {name: Species}; Species carries rqm = m/q, the charge q and
    # the mass m = rqm * q, all in OSIRIS units (electrons: rqm = -1, q = -1).
    for name, sp in deck.species.items():
        show(name, sp)
        show(f"  {name}.rqm / q / m", (sp.rqm, sp.q, sp.m))

    # Species objects can also be built by hand, e.g. for a synthetic diagnostic:
    show("Species('protons', 1836.0, q=1)", ou.Species("protons", 1836.0, q=1))

    # ------------------------------------------------------------------
    section("3. Reading parameters")
    # ------------------------------------------------------------------
    show("time_step / dt", deck.get_param("time_step", "dt"))
    show("time_step / ndump", deck.get_param("time_step", "ndump"))
    show("grid / nx_p", deck.get_param("grid", f"nx_p(1:{deck.dim})"))
    show("species / name (one per block)", deck.get_param("species", "name"))
    show("species / rqm", deck.get_param("species", "rqm"))

    # Values come back as the raw deck strings; `deval` handles Fortran floats
    # ('1.0d0' -> 1.0), which plain float() does not.
    tmax = deck.get_param("time", "tmax")[0]
    show("raw tmax string", repr(tmax))
    show("deval(tmax)", deval(tmax))
    dt0 = deck.get_param("time_step", "dt")[0]

    # __getitem__ gives the whole section as a dict (a deep copy — editing it
    # does not touch the deck).
    show("deck['time_step'][0]", deck["time_step"][0])
    show("missing section -> []", deck.get_param("does_not_exist", "x"))

    # ------------------------------------------------------------------
    section("4. Editing")
    # ------------------------------------------------------------------
    deck.set_param("time_step", "dt", 0.005)  # number
    deck.set_param("time", "tmax", 12.0)
    deck.set_param("grid", f"nx_p(1:{deck.dim})", [256] * deck.dim)  # list -> "256,256"
    deck.set_param("species", "name", "beam", i_use=0)  # str -> quoted, first block only
    show("dt now", deck.get_param("time_step", "dt"))
    show("nx_p now", deck.get_param("grid", f"nx_p(1:{deck.dim})"))
    show("species names now", deck.get_param("species", "name"))

    # i_use also takes a list of block indices.
    if deck.n_species > 1:
        deck.set_param("species", "rqm", -1.0, i_use=[0])
        show("rqm after i_use=[0]", deck.get_param("species", "rqm"))

    # A parameter that is not already in the section is refused unless you say
    # so — a typo would otherwise be written into a deck OSIRIS then rejects.
    try:
        deck.set_param("time_step", "ndump_fac", 2)
    except KeyError as e:
        show("set_param of a new key", f"KeyError: {e}")
    deck.set_param("time_step", "ndump_fac", 2, unexistent_ok=True)
    show("with unexistent_ok=True", deck.get_param("time_step", "ndump_fac"))

    deck.delete_param("time_step", "ndump_fac")
    show("after delete_param", "time_step" in dict(deck.sections) and "ndump_fac" not in deck["time_step"][0])

    # ------------------------------------------------------------------
    section("5. Templating a scan")
    # ------------------------------------------------------------------
    # set_tag replaces a placeholder everywhere it appears, which is the usual
    # way to drive a parameter scan from one template deck.  Note that
    # set_param() *quotes* a str value (it assumes you mean an OSIRIS string),
    # so a placeholder belongs in the template file itself, unquoted.
    args.outdir.mkdir(parents=True, exist_ok=True)
    template_file = args.outdir / "template.2d"
    template_file.write_text(deck_file.read_text().replace(f"dt = {deval(dt0)}", "dt = <DT>"))

    for dt in (0.01, 0.005):
        job = ou.InputDeckIO(str(template_file))
        job.set_tag("<DT>", dt)
        job.print_to_file(str(args.outdir / f"scan_dt{dt}.2d"))
        show(f"dt = {dt} ->", job.get_param("time_step", "dt"))

    # ------------------------------------------------------------------
    section("6. Writing the deck back out")
    # ------------------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "edited-deck.2d"
    deck.print_to_file(str(out))
    show("written to", out)
    print("\n".join(out.read_text().splitlines()[:14]))

    # And it round-trips: the written deck parses back to the same values.
    reread = ou.InputDeckIO(str(out))
    show("round-trip dt", reread.get_param("time_step", "dt"))

    print("\nDone.")


if __name__ == "__main__":
    main()

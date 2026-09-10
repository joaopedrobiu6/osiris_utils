import os
from pathlib import Path

import pytest

from osiris_utils.decks.decks import InputDeckIO
from osiris_utils.decks.species import Species

# --- Species Tests ---


def test_species_initialization():
    s = Species(name="electron", rqm=-1.0, q=-1)
    assert s.name == "electron"
    assert s.rqm == -1.0
    assert s.q == -1
    assert s.m == 1.0

    s2 = Species(name="proton", rqm=1.0 / 1836.0, q=1)
    assert s2.name == "proton"
    assert s2.rqm == 1.0 / 1836.0
    assert s2.q == 1
    assert s2.m == 1.0 / 1836.0


def test_species_defaults():
    s = Species(name="positron", rqm=1.0)
    # Default q is 1
    assert s.q == 1
    assert s.m == 1.0


@pytest.mark.parametrize(("rqm", "q_in"), [(-1.0, 1.0), (-1.0, -1.0), (-1.0, 2.0), (-1.0, -2.0)])
def test_species_takes_the_charge_sign_from_rqm(rqm, q_in):
    """Only |q| is taken from the argument; rqm = m/q (m > 0) fixes the sign.

    Otherwise q and rqm can disagree — and they did: ``Species(rqm=-1)`` used
    the default q = +1, so the density of an electron species came out positive.
    """
    s = Species(name="electron", rqm=rqm, q=q_in)
    assert s.q == -abs(q_in)
    assert s.m > 0
    assert s.m / s.q == pytest.approx(rqm)


def test_species_rejects_zero_charge():
    with pytest.raises(ValueError, match="q = 0"):
        Species(name="neutral", rqm=1.0, q=0)


# --- InputDeckIO Tests ---


@pytest.fixture
def sample_deck_file():
    return os.path.join(os.path.dirname(__file__), "data", "thermal.1d")


def test_input_deck_io_parsing(sample_deck_file):
    deck = InputDeckIO(sample_deck_file, verbose=True)
    assert deck.dim == 1  # nx_p(1:1) means 1D
    assert deck.n_species == 1

    # Check sections parsing
    sim_params = deck["time_step"][0]
    assert sim_params["dt"] == "0.0099"

    grid_params = deck["grid"][0]
    assert grid_params["nx_p(1:1)"] == "500"


def test_input_deck_io_species(sample_deck_file):
    deck = InputDeckIO(sample_deck_file)
    species = deck.species

    assert "electrons" in species
    assert len(species) == 1

    elec = species["electrons"]
    assert elec.name == "electrons"
    assert elec.rqm == -1.0
    # thermal.1d gives no q_real, so the magnitude defaults to 1; Species takes
    # the sign from rqm = -1.
    assert elec.q == -1


def test_input_deck_io_get_param(sample_deck_file):
    deck = InputDeckIO(sample_deck_file)
    val = deck.get_param("time_step", "dt")
    assert val == ["0.0099"]

    val = deck.get_param("species", "name")
    assert val == ['"electrons"']


def test_input_deck_io_set_param(sample_deck_file):
    deck = InputDeckIO(sample_deck_file)
    deck.set_param("time_step", "dt", "0.1")
    assert deck.get_param("time_step", "dt") == ['"0.1"']

    # Test setting numeric
    deck.set_param("time_step", "dt", 0.2)
    assert deck.get_param("time_step", "dt") == ["0.2"]


def test_input_deck_io_write(sample_deck_file, tmp_path):
    deck = InputDeckIO(sample_deck_file)
    deck.set_param("time_step", "dt", 0.99)

    out_file = tmp_path / "new.input"
    deck.print_to_file(str(out_file))

    # Read back
    deck2 = InputDeckIO(str(out_file))
    assert deck2.get_param("time_step", "dt") == ["0.99"]


def test_species_q_real_is_read_per_species(tmp_path, sample_deck_file):
    """Each species gets its OWN q_real.

    ``s_qreal[0]`` handed every species the first one's charge, so in a
    two-species deck the ions inherited the electrons' q and their mass
    (m = rqm * q) came out wrong with it.
    """
    src = Path(sample_deck_file).read_text()
    src = src.replace("num_species = 1,", "num_species = 2,")
    electrons = src[src.index("species\n{") : src.index("}", src.index("species\n{")) + 2]
    src = src.replace(
        electrons,
        electrons.replace("\trqm = -1.0,\n", "\trqm = -1.0,\n\tq_real = 1.0,\n")
        + '\nspecies\n{\n\tname = "ions",\n\trqm = 16.0,\n\tq_real = 2.0,\n\tnum_par_x(1:1) = 64,\n}\n',
        1,
    )
    deck_path = tmp_path / "two_species.1d"
    deck_path.write_text(src)

    species = InputDeckIO(str(deck_path)).species
    assert species["electrons"].q == -1.0
    assert species["ions"].q == 2.0  # its own q_real, not the electrons' 1.0
    assert species["ions"].m == 32.0  # m = rqm * q = 16 * 2
    for sp in species.values():
        assert sp.m / sp.q == pytest.approx(sp.rqm)

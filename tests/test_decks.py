import os

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


# --- InputDeckIO Tests ---


@pytest.fixture
def sample_deck_file():
    return os.path.join(os.path.dirname(__file__), "../examples/example_data/thermal.1d")


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
    # No q_real specified in thermal.1d for electrons.
    # Code: if q_real not provided, assumes ones.
    # q calculation: q = int(s_qreal[0]) * np.sign(float(s_rqm[i]))
    # s_qreal defaults to np.ones(len(s_names)). So s_qreal[0] = 1.
    # rqm = -1.0. sign is -1.
    # q = 1 * -1 = -1.
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

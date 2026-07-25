from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.data import OsirisTrackFile, get_track_indexes, reorder_track_data
from osiris_utils.data.track_diagnostic import Track_Diagnostic

from .conftest import (
    DT,
    NDUMP,
    SPECIES,
    TRACK_LABELS,
    TRACK_N_ITERS,
    TRACK_N_PARTICLES,
    TRACK_QUANTS,
    TRACK_UNITS,
    XMAX,
    XMIN,
    track_value,
)


def _read_file_tags(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines()[5:]]
    return np.array(rows, dtype=int)


def _expected(quant: str) -> np.ndarray:
    """Reference (n_particles, n_iters) array for a tracked quantity."""
    return np.array(
        [[track_value(quant, p, k) for k in range(TRACK_N_ITERS)] for p in range(1, TRACK_N_PARTICLES + 1)],
        dtype=float,
    )


def test_track_index_helpers_reorder_idl_track_data() -> None:
    itermap = np.array(
        [
            [1, 2, 0],
            [2, 1, 0],
            [1, 1, 2],
            [2, 2, 1],
        ],
        dtype=np.int64,
    )
    unordered_data = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [0.0, 20.0],
            [2.0, 12.0],
            [1.0, 21.0],
            [2.0, 22.0],
        ]
    )

    indexes = get_track_indexes(itermap, num_particles=2)
    assert indexes == [[0, 1, 3], [2, 4, 5]]

    ordered = reorder_track_data(unordered_data, indexes, ["t", "x1"])
    assert ordered.shape == (2, 3)
    assert ordered.dtype.names == ("t", "x1")
    np.testing.assert_allclose(ordered["t"], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    np.testing.assert_allclose(ordered["x1"], [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])


def test_osiris_track_file_reads_track_data(track_path: Path) -> None:
    track = OsirisTrackFile(str(track_path))

    assert track.name == SPECIES
    assert track.type == "tracks-2"
    assert track.dim == 1
    assert track.dt == pytest.approx(DT)
    assert track.num_particles == TRACK_N_PARTICLES
    assert track.num_time_iters == TRACK_N_ITERS
    assert track.data.shape == (TRACK_N_PARTICLES, TRACK_N_ITERS)
    assert track.quants == TRACK_QUANTS[1:]
    assert track.units["x1"] == TRACK_UNITS["x1"]
    assert track.labels["p1"] == TRACK_LABELS["p1"]
    np.testing.assert_allclose(track.grid, [[XMIN, XMAX]])

    np.testing.assert_allclose(track.data["t"][0], np.arange(TRACK_N_ITERS) * DT)


def test_osiris_track_file_reorders_interleaved_chunks(track_path: Path) -> None:
    """Particles are written in interleaved chunks; reading must de-interleave them."""
    track = OsirisTrackFile(str(track_path))

    for quant in ["t", "x1", "p1", "ene"]:
        np.testing.assert_allclose(track.data[quant], _expected(quant), rtol=1e-9)


def test_track_diagnostic_lazy_access_and_load_all(sim_dir: Path, track_path: Path) -> None:
    deck = ou.InputDeckIO(str(sim_dir / "thermal.1d"), verbose=False)
    tracks = Track_Diagnostic(str(sim_dir), species=ou.Species(SPECIES, -1), input_deck=deck)
    raw_track = OsirisTrackFile(str(track_path))

    assert tracks.path == str(track_path)
    assert tracks.quantity == "tracks"
    assert tracks.ndump == NDUMP
    assert tracks.num_particles == raw_track.num_particles
    assert tracks.num_time_iters == raw_track.num_time_iters
    assert tracks.quants == raw_track.quants

    np.testing.assert_allclose(tracks["p1"][0:4, 5], raw_track.data["p1"][0:4, 5])
    with pytest.raises(ValueError, match="Data not loaded"):
        _ = tracks.time

    loaded = tracks.load_all()
    assert loaded is tracks.data
    np.testing.assert_allclose(tracks.time, raw_track.data["t"][0])
    np.testing.assert_allclose(tracks["x1"][0], raw_track.data["x1"][0])

    tracks.unload()
    with pytest.raises(ValueError, match="Data not loaded"):
        _ = tracks.data


def test_simulation_species_tracks_uses_track_diagnostic(sim_dir: Path, track_path: Path) -> None:
    sim = ou.Simulation(str(sim_dir / "thermal.1d"))

    with pytest.raises(ValueError, match="Tracks diagnostics require a specie"):
        _ = sim["tracks"]

    species = sim[SPECIES]
    tracks = species["tracks"]
    raw_track = OsirisTrackFile(str(track_path))

    assert isinstance(tracks, Track_Diagnostic)
    assert tracks.path == str(track_path)
    assert "tracks" not in species.loaded_diagnostics
    np.testing.assert_allclose(tracks["p1"][0:4, 5], raw_track.data["p1"][0:4, 5])

    loaded_tracks = tracks.load_all()
    assert loaded_tracks is tracks
    assert species.loaded_diagnostics["tracks"] is tracks
    assert species["tracks"] is tracks
    np.testing.assert_allclose(tracks.data["p1"][0:4, 5], raw_track.data["p1"][0:4, 5])


def test_raw_to_file_tags_from_simulation_tree(raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(raw_path))

    assert {"x1", "p1", "p2", "p3", "q", "ene", "tag"} <= set(raw.data)
    assert raw.labels["p1"] == "p_1"
    assert raw.units["p1"] == "m_e c"

    random_tags_path = tmp_path / "random_file_tags.tags"
    raw.raw_to_file_tags(str(random_tags_path), type="random", n_tags=10)
    random_tags = _read_file_tags(random_tags_path)
    assert random_tags.shape == (10, 2)
    available_tags = {tuple(row) for row in np.abs(raw.data["tag"])}
    assert set(map(tuple, random_tags)) <= available_tags

    mask = raw.data["p1"] > 0.025
    masked_tags_path = tmp_path / "masked_file_tags.tags"
    raw.raw_to_file_tags(str(masked_tags_path), type="all", mask=mask)
    masked_tags = _read_file_tags(masked_tags_path)
    expected_tags = np.abs(raw.data["tag"][mask])
    expected_tags = expected_tags[np.lexsort((expected_tags[:, 1], expected_tags[:, 0]))]

    assert masked_tags.shape == (int(mask.sum()), 2)
    np.testing.assert_array_equal(masked_tags, expected_tags)


def test_convert_tracks_writes_v2_layout(track_path: Path, tmp_path: Path) -> None:
    track_copy = tmp_path / f"{SPECIES}-tracks.h5"
    shutil.copyfile(track_path, track_copy)

    converted_path = Path(ou.convert_tracks(str(track_copy)))
    assert converted_path == tmp_path / f"{SPECIES}-tracks-v2.h5"
    assert converted_path.exists()

    track = OsirisTrackFile(str(track_copy))
    with h5py.File(converted_path, "r") as file:
        assert "1" in file
        assert str(TRACK_N_PARTICLES) in file
        np.testing.assert_array_equal(file["1"][b"n"][:4], [0, 1, 2, 3])
        for quant in ["t", "x1", "p1"]:
            np.testing.assert_allclose(file["1"][quant.encode()][:], track.data[quant][0])

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.data import OsirisTrackFile, get_track_indexes, reorder_track_data
from osiris_utils.data.track_diagnostic import Track_Diagnostic


@pytest.fixture
def example_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "example_data"


@pytest.fixture
def example_track_path(example_data_dir: Path) -> Path:
    return example_data_dir / "MS" / "TRACKS" / "electrons-tracks.h5"


def _read_file_tags(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines()[5:]]
    return np.array(rows, dtype=int)


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


def test_osiris_track_file_reads_example_track_data(example_track_path: Path) -> None:
    track = OsirisTrackFile(str(example_track_path))

    assert track.name == "electrons"
    assert track.type == "tracks-2"
    assert track.dim == 1
    assert track.dt == pytest.approx(0.0099)
    assert track.num_particles == 229
    assert track.num_time_iters == 241
    assert track.data.shape == (track.num_particles, track.num_time_iters)
    assert track.quants[:5] == ["t", "q", "ene", "x1", "p1"]
    assert track.units["x1"] == "c/\\omega_p"
    assert track.labels["p1"] == "p_1"
    np.testing.assert_allclose(track.grid, [[0.0, 5.0]])
    np.testing.assert_allclose(track.data["t"][0, :4], [0.0, 0.0099, 0.0198, 0.0297])

    with h5py.File(example_track_path, "r") as file:
        np.testing.assert_allclose(track.data[0, 0].tolist(), file["data"][0])
        np.testing.assert_allclose(track.data[1, 0].tolist(), file["data"][21])


def test_track_diagnostic_lazy_access_and_load_all(example_data_dir: Path, example_track_path: Path) -> None:
    deck = ou.InputDeckIO(str(example_data_dir / "thermal.1d"), verbose=False)
    tracks = Track_Diagnostic(str(example_data_dir), species=ou.Species("electrons", -1), input_deck=deck)
    raw_track = OsirisTrackFile(str(example_track_path))

    assert tracks.path == str(example_track_path)
    assert tracks.quantity == "tracks"
    assert tracks.ndump == 20
    assert tracks.num_particles == raw_track.num_particles
    assert tracks.num_time_iters == raw_track.num_time_iters
    assert tracks.quants == raw_track.quants

    np.testing.assert_allclose(tracks["p1"][0:4, 50], raw_track.data["p1"][0:4, 50])
    with pytest.raises(ValueError, match="Data not loaded"):
        _ = tracks.time

    loaded = tracks.load_all()
    assert loaded is tracks.data
    np.testing.assert_allclose(tracks.time, raw_track.data["t"][0])
    np.testing.assert_allclose(tracks["x1"][0], raw_track.data["x1"][0])

    tracks.unload()
    with pytest.raises(ValueError, match="Data not loaded"):
        _ = tracks.data


def test_simulation_species_tracks_uses_track_diagnostic(example_data_dir: Path, example_track_path: Path) -> None:
    sim = ou.Simulation(str(example_data_dir / "thermal.1d"))

    with pytest.raises(ValueError, match="Tracks diagnostics require a specie"):
        _ = sim["tracks"]

    species = sim["electrons"]
    tracks = species["tracks"]
    raw_track = OsirisTrackFile(str(example_track_path))

    assert isinstance(tracks, Track_Diagnostic)
    assert tracks.path == str(example_track_path)
    assert "tracks" not in species.loaded_diagnostics
    np.testing.assert_allclose(tracks["p1"][0:4, 50], raw_track.data["p1"][0:4, 50])

    loaded_tracks = tracks.load_all()
    assert loaded_tracks is tracks
    assert species.loaded_diagnostics["tracks"] is tracks
    assert species["tracks"] is tracks
    np.testing.assert_allclose(tracks.data["p1"][0:4, 50], raw_track.data["p1"][0:4, 50])


def test_example_raw_tracks_notebook_raw_to_file_tags(example_data_dir: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(example_data_dir / "MS" / "RAW" / "electrons" / "RAW-electrons-000050.h5"))

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


def test_example_raw_tracks_notebook_convert_tracks(example_track_path: Path, tmp_path: Path) -> None:
    track_copy = tmp_path / "electrons-tracks.h5"
    shutil.copyfile(example_track_path, track_copy)

    converted_path = Path(ou.utils.convert_tracks(str(track_copy)))
    assert converted_path == tmp_path / "electrons-tracks-v2.h5"
    assert converted_path.exists()

    track = OsirisTrackFile(str(track_copy))
    with h5py.File(converted_path, "r") as file:
        assert "1" in file
        assert "229" in file
        np.testing.assert_array_equal(file["1"][b"n"][:4], [0, 1, 2, 3])
        np.testing.assert_allclose(file["1"][b"t"][:4], track.data["t"][0, :4])
        np.testing.assert_allclose(file["1"][b"x1"][:4], track.data["x1"][0, :4])
        np.testing.assert_allclose(file["1"][b"p1"][:4], track.data["p1"][0, :4])

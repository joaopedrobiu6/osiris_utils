from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.data import OsirisRawFile


@pytest.fixture
def example_raw_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "example_data" / "MS" / "RAW" / "electrons" / "RAW-electrons-000050.h5"


def _read_file_tags(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines()[5:]]
    return np.array(rows, dtype=int)


def test_osiris_raw_file_reads_example_particle_data(example_raw_path: Path) -> None:
    raw = OsirisRawFile(str(example_raw_path))

    assert raw.name == "electrons"
    assert raw.type == "particles"
    assert raw.dim == 1
    assert raw.dt == pytest.approx(0.0099)
    assert raw.iter == 50
    assert raw.time == [pytest.approx(0.495), "1 / \\omega_p"]
    np.testing.assert_allclose(raw.grid, [[0.0, 5.0]])

    assert raw.quants == ["x1", "p1", "p2", "p3", "q", "ene", "tag"]
    assert raw.labels == {
        "x1": "x_1",
        "p1": "p_1",
        "p2": "p_2",
        "p3": "p_3",
        "q": "q",
        "ene": "Ene",
        "tag": "Tag",
    }
    assert raw.units["x1"] == "c/\\omega_p"
    assert raw.units["p1"] == "m_e c"
    assert raw.units["ene"] == "m_e c^2"
    assert raw.units["tag"] == ""

    for key in ["x1", "p1", "p2", "p3", "q", "ene"]:
        assert raw.data[key].shape == (32000,)
        assert raw.data[key].dtype == np.float32

    assert raw.data["tag"].shape == (32000, 2)
    assert raw.data["tag"].dtype == np.int32
    np.testing.assert_allclose(raw.data["x1"][:3], [0.00935326, 0.00459187, 0.00702063], rtol=1e-6)
    np.testing.assert_allclose(raw.data["p1"][:3], [0.02101514, 0.00883643, 0.02322492], rtol=1e-6)
    np.testing.assert_array_equal(raw.data["tag"][:3], [[4, 7994], [1, 2], [4, 7972]])


def test_osiris_raw_file_axis_metadata_matches_quantities(example_raw_path: Path) -> None:
    raw = OsirisRawFile(str(example_raw_path))

    assert set(raw.axis) == set(raw.quants)
    assert raw.axis["x1"] == {
        "name": "x1",
        "units": "c/\\omega_p",
        "long_name": "x_1",
    }
    assert raw.axis["tag"] == {
        "name": "tag",
        "units": "",
        "long_name": "Tag",
    }


def test_raw_to_file_tags_writes_all_tags_sorted_and_positive(example_raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(example_raw_path))
    tags_path = tmp_path / "all_file_tags.tags"

    raw.raw_to_file_tags(str(tags_path), type="all")

    written_tags = _read_file_tags(tags_path)
    expected_tags = np.abs(raw.data["tag"])
    expected_tags = expected_tags[np.lexsort((expected_tags[:, 1], expected_tags[:, 0]))]

    assert written_tags.shape == (32000, 2)
    np.testing.assert_array_equal(written_tags, expected_tags)


def test_raw_to_file_tags_writes_masked_tags(example_raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(example_raw_path))
    tags_path = tmp_path / "masked_file_tags.tags"
    mask = raw.data["p1"] > 0.025

    raw.raw_to_file_tags(str(tags_path), type="all", mask=mask)

    written_tags = _read_file_tags(tags_path)
    expected_tags = np.abs(raw.data["tag"][mask])
    expected_tags = expected_tags[np.lexsort((expected_tags[:, 1], expected_tags[:, 0]))]

    assert written_tags.shape == (229, 2)
    np.testing.assert_array_equal(written_tags, expected_tags)


def test_raw_to_file_tags_random_selection_uses_available_tags(example_raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(example_raw_path))
    tags_path = tmp_path / "random_file_tags.tags"

    raw.raw_to_file_tags(str(tags_path), type="random", n_tags=10)

    written_tags = _read_file_tags(tags_path)
    available_tags = {tuple(row) for row in np.abs(raw.data["tag"])}

    assert written_tags.shape == (10, 2)
    assert set(map(tuple, written_tags)) <= available_tags


def test_raw_to_file_tags_validates_mask_and_selection(example_raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(example_raw_path))

    with pytest.raises(ValueError, match="Mask must be"):
        raw.raw_to_file_tags(str(tmp_path / "bad_mask.tags"), mask=np.ones(10, dtype=bool))

    with pytest.raises(ValueError, match="Not enough tags"):
        raw.raw_to_file_tags(str(tmp_path / "too_many.tags"), type="random", n_tags=raw.data["tag"].shape[0] + 1)

    with pytest.raises(TypeError, match="Invalid type"):
        raw.raw_to_file_tags(str(tmp_path / "bad_type.tags"), type="first")

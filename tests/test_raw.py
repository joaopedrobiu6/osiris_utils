from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.data import OsirisRawFile

from .conftest import (
    DT,
    RAW_ITER,
    RAW_LABELS,
    RAW_N_PARTICLES,
    RAW_QUANTS,
    RAW_UNITS,
    SPECIES,
    XMAX,
    XMIN,
    raw_values,
)


def _read_file_tags(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines()[5:]]
    return np.array(rows, dtype=int)


def _expected_sorted_tags(tags: np.ndarray) -> np.ndarray:
    tags = np.abs(tags)
    return tags[np.lexsort((tags[:, 1], tags[:, 0]))]


def test_osiris_raw_file_reads_particle_data(raw_path: Path) -> None:
    raw = OsirisRawFile(str(raw_path))

    assert raw.name == SPECIES
    assert raw.type == "particles"
    assert raw.dim == 1
    assert raw.dt == pytest.approx(DT)
    assert raw.iter == RAW_ITER
    assert raw.time == [pytest.approx(RAW_ITER * DT), "1 / \\omega_p"]
    np.testing.assert_allclose(raw.grid, [[XMIN, XMAX]])

    assert raw.quants == RAW_QUANTS
    assert raw.labels == RAW_LABELS
    assert raw.units == RAW_UNITS

    for key in ["x1", "p1", "p2", "p3", "q", "ene"]:
        assert raw.data[key].shape == (RAW_N_PARTICLES,)
        assert raw.data[key].dtype == np.float32
        np.testing.assert_allclose(raw.data[key], raw_values(key), rtol=1e-6)

    assert raw.data["tag"].shape == (RAW_N_PARTICLES, 2)
    assert raw.data["tag"].dtype == np.int32
    np.testing.assert_array_equal(raw.data["tag"], raw_values("tag"))


def test_osiris_raw_file_closes_its_hdf5_handle(raw_path: Path) -> None:
    """Reading many RAW dumps must not leak file descriptors."""
    import h5py

    def open_files() -> int:
        return h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)

    before = open_files()
    readers = [OsirisRawFile(str(raw_path)) for _ in range(5)]

    assert open_files() == before, "OsirisRawFile leaked open HDF5 handles"
    assert all(not r._file.id.valid for r in readers)


def test_osiris_raw_file_data_survives_closing_the_handle(raw_path: Path) -> None:
    """Closing must happen only after every dataset has been materialised."""
    raw = OsirisRawFile(str(raw_path))

    for quant in RAW_QUANTS:
        np.testing.assert_array_equal(raw.data[quant], raw_values(quant))


def test_osiris_raw_file_axis_metadata_matches_quantities(raw_path: Path) -> None:
    raw = OsirisRawFile(str(raw_path))

    assert set(raw.axis) == set(raw.quants)
    for quant in RAW_QUANTS:
        assert raw.axis[quant] == {
            "name": quant,
            "units": RAW_UNITS[quant],
            "long_name": RAW_LABELS[quant],
        }


def test_raw_to_file_tags_writes_all_tags_sorted_and_positive(raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(raw_path))
    tags_path = tmp_path / "all_file_tags.tags"

    raw.raw_to_file_tags(str(tags_path), type="all")

    written_tags = _read_file_tags(tags_path)

    assert written_tags.shape == (RAW_N_PARTICLES, 2)
    assert (written_tags > 0).all()
    np.testing.assert_array_equal(written_tags, _expected_sorted_tags(raw_values("tag")))


def test_raw_to_file_tags_writes_masked_tags(raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(raw_path))
    tags_path = tmp_path / "masked_file_tags.tags"
    mask = raw.data["p1"] > 0.025

    raw.raw_to_file_tags(str(tags_path), type="all", mask=mask)

    written_tags = _read_file_tags(tags_path)

    assert mask.sum() > 0, "mask must select a non-trivial subset"
    assert written_tags.shape == (int(mask.sum()), 2)
    np.testing.assert_array_equal(written_tags, _expected_sorted_tags(raw_values("tag")[mask]))


def test_raw_to_file_tags_random_selection_uses_available_tags(raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(raw_path))
    tags_path = tmp_path / "random_file_tags.tags"

    raw.raw_to_file_tags(str(tags_path), type="random", n_tags=10)

    written_tags = _read_file_tags(tags_path)
    available_tags = {tuple(row) for row in np.abs(raw_values("tag"))}

    assert written_tags.shape == (10, 2)
    assert set(map(tuple, written_tags)) <= available_tags


def test_raw_to_file_tags_validates_mask_and_selection(raw_path: Path, tmp_path: Path) -> None:
    raw = ou.OsirisRawFile(str(raw_path))

    with pytest.raises(ValueError, match="Mask must be"):
        raw.raw_to_file_tags(str(tmp_path / "bad_mask.tags"), mask=np.ones(10, dtype=bool))

    with pytest.raises(ValueError, match="Not enough tags"):
        raw.raw_to_file_tags(str(tmp_path / "too_many.tags"), type="random", n_tags=RAW_N_PARTICLES + 1)

    with pytest.raises(TypeError, match="Invalid type"):
        raw.raw_to_file_tags(str(tmp_path / "bad_type.tags"), type="first")

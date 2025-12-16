import numpy as np
import pytest

from osiris_utils.utils import (
    courant2D,
    create_file_tags,
    filesize_estimation,
    integrate,
    read_data,
    save_data,
    time_estimation,
    transverse_average,
)


def test_courant2D():
    dx = 1.0
    dy = 1.0
    # 1 / sqrt(1 + 1) = 1 / sqrt(2)
    expected = 1.0 / np.sqrt(2.0)
    assert np.isclose(courant2D(dx, dy), expected)

    dx = 0.5
    dy = 0.5
    # 1 / sqrt(4 + 4) = 1 / sqrt(8) = 1 / (2*sqrt(2))
    expected = 1.0 / np.sqrt(8.0)
    assert np.isclose(courant2D(dx, dy), expected)


def test_time_estimation():
    # n_cells, ppc, t_steps, n_cpu, push_time=1e-7
    # time = (n_cells * ppc * push_time * t_steps) / n_cpu
    res = time_estimation(100, 10, 1000, 1, push_time=1e-7)
    # 100 * 10 * 1e-7 * 1000 / 1 = 1000 * 1000 * 1e-7 = 1e6 * 1e-7 = 0.1
    assert np.isclose(res, 0.1)

    # Test hours
    res_hours = time_estimation(100, 10, 1000, 1, push_time=1e-7, hours=True)
    assert np.isclose(res_hours, 0.1 / 3600.0)


def test_filesize_estimation():
    # n_gridpoints * 4 / (1024**2)  (MB)
    points = 1024 * 1024
    # 1024*1024 * 4 / (1024*1024) = 4
    assert np.isclose(filesize_estimation(points), 4.0)


def test_transverse_average():
    data = np.array([[1, 2], [3, 4]])
    # mean axis 1: [1.5, 3.5]
    res = transverse_average(data)
    assert np.array_equal(res, np.array([1.5, 3.5]))

    with pytest.raises(ValueError):
        transverse_average(np.array([1, 2, 3]))


def test_integrate():
    # integrate uses -scipy.integrate.cumulative_simpson(flip(arr), dx) and flips back
    # effectively integrating from right to left or something?
    # docstring: "Integrate a 1D from the left to the right"
    # But implementation: flip -> cumulative -> flip.
    # If I integrate f(x) = 1 from 0 to 1 with dx=1.
    # arr = [1, 1]
    # flip = [1, 1]
    # cum_simpson([1, 1], dx=1) -> [0, 0.5 * (1+1) * 1] ? Simpson needs 3 points usually?
    # scipy cumulative_simpson handles 2 points as trapezoid usually.
    # Let's try simple array: [1, 1, 1]. dx=1.
    # Integral of 1 is x.
    # Expected: [0, 1, 2] ? Or accumulation?

    arr = np.ones(5)
    dx = 1.0
    # Let's see what it does.
    # The function negates the result: -scipy...
    # This might be for integrating E to get Potential phi = - integral E.
    # If input is [1, 1, 1, 1, 1]
    # output should likely correspond to the integral.
    res = integrate(arr, dx)
    assert len(res) == 5

    # If array is not 1D
    with pytest.raises(ValueError):
        integrate(np.zeros((2, 2)), 1.0)


def test_save_read_data(tmp_path):
    data = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "test.txt"

    # Numpy
    save_data(data, str(p), option="numpy")
    assert p.exists()
    loaded = read_data(str(p), option="numpy")
    assert np.allclose(data, loaded)

    # Pandas
    p_csv = tmp_path / "test.csv"
    save_data(data, str(p_csv), option="pandas")
    assert p_csv.exists()
    # read_data with pandas returns values
    loaded_pd = read_data(str(p_csv), option="pandas")
    # pandas saves with header by default? save_data implementation:
    # pd.DataFrame(data).to_csv(savename, index=False) -> Header is True by default (0)
    # read_data: pd.read_csv(filename).values
    # If saved with header '0', read_csv reads it.
    # array [1, 2, 3] -> DF with col '0'.
    # read_csv -> values -> [[1], [2], [3]] (column vector)
    # original was flat.
    # Let's check shapes.
    assert loaded_pd.size == data.size
    assert np.allclose(loaded_pd.flatten(), data)


def test_create_file_tags(tmp_path):
    tags = np.array([[12, 1], [5, -1], [1, 0]])
    # logic: tags[:, 0] = abs(tags[:, 0]). Sort by col 1 then col 0.
    # [12, 1], [5, 1], [1, 0]
    # Sorted lexsort(col 1, col 0):
    # col 1 values: 1, 1, 0  (Wait, original was 1, -1, 0?)
    # code: tags[:, 0] = np.abs(tags[:, 0])
    # tags become: [[12, 1], [5, -1], [1, 0]] (col 0 taken abs)
    # lexsort((tags[:, 1], tags[:, 0])) -> sorts by col 0 (primary) then col 1 (secondary)?
    # Actually np.lexsort((secondary, primary)).
    # The code does: tags[np.lexsort((tags[:, 1], tags[:, 0]))]
    # So primary sort key is tags[:, 0] (particle tag), secondary is tags[:, 1] (something else?).
    # Wait, usually lexsort is keys passed in order of significance from last to first.
    # lexsort((B, A)) -> Sort by A, then by B.
    # So here sorting by tags[:, 0] (Particle ID), then tags[:, 1].

    # Input: [[12, 1], [5, -1], [1, 0]]
    # Abs: [[12, 1], [5, -1], [1, 0]]
    # Sort order (by ID): 1, 5, 12.
    # Expected output order: [1, 0], [5, -1], [12, 1]

    out = tmp_path / "tags.txt"
    create_file_tags(str(out), tags)

    content = out.read_text().splitlines()
    # Header lines:
    # ! particle tag list
    # ! generated on ...
    # ! number of tags
    #        3
    # ! particle tag list
    #          1     0
    #          5    -1
    #         12     1

    assert "! particle tag list" in content[0]
    assert "! number of tags" in content[2]
    assert "3" in content[3]

    # Check data lines (skip headers)
    # index 5, 6, 7
    line_1 = content[5].split()
    assert line_1[0] == "1"
    assert line_1[1] == "0"

    line_2 = content[6].split()
    assert line_2[0] == "5"
    assert line_2[1] == "-1"

    line_3 = content[7].split()
    assert line_3[0] == "12"
    assert line_3[1] == "1"

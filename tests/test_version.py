import osiris_utils as ou


def test_version_string():
    assert isinstance(ou.__version__, str)
    assert ou.__version__ not in ("", "0.0.0")

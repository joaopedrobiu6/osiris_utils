import osiris_utils as ou


def test_version_string():
    assert isinstance(ou.__version__, str) and ou.__version__

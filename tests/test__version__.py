"""Unit tests for `gempyor2.__version__`."""

from gempyor2 import __version__


def test_version_is_string() -> None:
    """Test that `__version__` is a string."""
    assert isinstance(__version__, str)

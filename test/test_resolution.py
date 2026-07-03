"""Tests for zivid.Resolution class."""

import pytest
import zivid


def test_resolution_init():
    """Test creating Resolution with width and height."""
    resolution = zivid.Resolution(2448, 2048)

    assert resolution.width == 2448
    assert resolution.height == 2048
    assert resolution.size == 2448 * 2048


def test_resolution_init_zero_dimensions():
    """Test creating Resolution with zero dimensions."""
    resolution = zivid.Resolution(0, 0)

    assert resolution.width == 0
    assert resolution.height == 0
    assert resolution.size == 0


def test_resolution_init_invalid_types():
    """Test error handling for invalid types."""
    with pytest.raises(TypeError, match="Width and height must be integers"):
        zivid.Resolution(2448.5, 2048)

    with pytest.raises(TypeError, match="Width and height must be integers"):
        zivid.Resolution("2448", 2048)

    with pytest.raises(TypeError, match="Width and height must be integers"):
        zivid.Resolution(2448, "2048")


def test_resolution_init_negative_values():
    """Test error handling for negative values."""
    with pytest.raises(ValueError, match="Width and height must be non-negative"):
        zivid.Resolution(-1, 2048)

    with pytest.raises(ValueError, match="Width and height must be non-negative"):
        zivid.Resolution(2448, -1)

    with pytest.raises(ValueError, match="Width and height must be non-negative"):
        zivid.Resolution(-1, -1)


def test_resolution_properties():
    """Test Resolution properties."""
    res1 = zivid.Resolution(2448, 2048)

    # Test properties are read-only and correct
    assert res1.width == 2448
    assert res1.height == 2048
    assert res1.size == 2448 * 2048


def test_resolution_equality():
    """Test Resolution equality comparison."""
    res1 = zivid.Resolution(2448, 2048)
    res2 = zivid.Resolution(2448, 2048)
    res3 = zivid.Resolution(2816, 2816)

    # Test equality
    assert res1 == res2
    assert res1 != res3
    assert res2 != res3

    # Test with non-Resolution objects
    assert res1 != "2448x2048"
    assert res1 != (2448, 2048)
    assert res1 is not None


def test_resolution_string_representation():
    """Test string representation of Resolution."""
    resolution = zivid.Resolution(1944, 1200)

    str_repr = str(resolution)
    repr_repr = repr(resolution)

    # Check str format (from C++ toString())
    assert "1944" in str_repr
    assert "1200" in str_repr

    # Check repr format
    assert repr_repr == "Resolution(width=1944, height=1200)"


def test_resolution_hash():
    """Test Resolution hashing for use in sets/dicts."""
    res1 = zivid.Resolution(2448, 2048)
    res2 = zivid.Resolution(2448, 2048)
    res3 = zivid.Resolution(2816, 2816)

    # Equal resolutions should have equal hashes
    assert hash(res1) == hash(res2)

    # Can be used in sets
    resolution_set = {res1, res2, res3}
    assert len(resolution_set) == 2  # res1 and res2 are the same

    # Can be used as dict keys
    resolution_dict = {res1: "Full HD", res3: "HD"}
    assert len(resolution_dict) == 2
    assert resolution_dict[res2] == "Full HD"  # res2 == res1


def test_resolution_immutability():
    """Test that Resolution properties cannot be modified."""
    resolution = zivid.Resolution(2448, 2048)

    # Properties should be read-only
    with pytest.raises(AttributeError):
        resolution.width = 2448  # pylint: disable=attribute-defined-outside-init

    with pytest.raises(AttributeError):
        resolution.height = 2048  # pylint: disable=attribute-defined-outside-init

    with pytest.raises(AttributeError):
        resolution.size = 2448 * 2048  # pylint: disable=attribute-defined-outside-init

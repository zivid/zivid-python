"""Tests for zivid.Mask class."""

import numpy as np
import pytest
import zivid


def test_mask_init_from_numpy_boolean():
    """Test creating Mask from numpy boolean array."""
    mask_data = np.array([[True, False, True], [False, True, False]], dtype=bool)

    mask = zivid.Mask(mask_data)

    assert mask.width == 3
    assert mask.height == 2

    # Convert back to array and check values
    result_array = mask.to_array()
    expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(result_array, expected)


def test_mask_init_from_numpy_uint8():
    """Test creating Mask from numpy uint8 array."""
    mask_data = np.array([[1, 0, 255], [0, 128, 0]], dtype=np.uint8)

    mask = zivid.Mask(mask_data)

    assert mask.width == 3
    assert mask.height == 2

    # Should preserve uint8 values exactly
    result_array = mask.to_array()
    np.testing.assert_array_equal(result_array, mask_data)


def test_mask_init_from_numpy_int():
    """Test creating Mask from numpy integer array."""
    mask_data = np.array([[1, 0, -1], [0, 5, 0]], dtype=int)

    mask = zivid.Mask(mask_data)

    # Should convert non-zero to 1, zero to 0
    result_array = mask.to_array()
    expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(result_array, expected)


def test_mask_init_from_numpy_float():
    """Test creating Mask from numpy float array."""
    mask_data = np.array([[1.5, 0.0, -0.1], [0.0, 0.001, 0.0]], dtype=float)

    mask = zivid.Mask(mask_data)

    # Should convert non-zero to 1, zero to 0
    result_array = mask.to_array()
    expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(result_array, expected)


def test_mask_init_from_resolution():
    """Test creating Mask from Resolution object."""
    resolution = zivid.Resolution(width=5, height=3)

    mask = zivid.Mask(resolution)

    assert mask.width == 5
    assert mask.height == 3

    # Should be filled with ones (no masking)
    result_array = mask.to_array()
    expected = np.ones((3, 5), dtype=np.uint8)

    np.testing.assert_array_equal(result_array, expected)


def test_mask_properties():
    """Test Mask properties."""
    mask_data = np.ones((10, 15), dtype=bool)
    mask = zivid.Mask(mask_data)

    assert mask.width == 15
    assert mask.height == 10

    resolution = mask.resolution
    assert resolution.width == 15
    assert resolution.height == 10


def test_mask_string_representation():
    """Test string representation of Mask."""
    mask_data = np.ones((2, 3), dtype=bool)
    mask = zivid.Mask(mask_data)

    str_repr = str(mask)
    repr_repr = repr(mask)

    # Should contain dimensions
    assert "Width" in str_repr or "width" in str_repr.lower()
    assert "Height" in str_repr or "height" in str_repr.lower()
    assert "3" in str_repr  # width
    assert "2" in str_repr  # height

    # __str__ and __repr__ should be the same
    assert str_repr == repr_repr


def test_mask_invalid_dimensions():
    """Test error handling for invalid array dimensions."""
    # 1D array should fail
    with pytest.raises(ValueError, match="2D array"):
        zivid.Mask(np.array([1, 0, 1]))

    # 3D array should fail
    with pytest.raises(ValueError, match="2D array"):
        zivid.Mask(np.ones((2, 3, 4)))


def test_mask_invalid_type():
    """Test error handling for invalid input types."""
    with pytest.raises(TypeError):
        zivid.Mask("invalid")

    with pytest.raises(TypeError):
        zivid.Mask([1, 2, 3])

    with pytest.raises(TypeError):
        zivid.Mask(123)


def test_mask_empty_array():
    """Test creating Mask from empty array."""
    # Should handle empty arrays gracefully
    empty_array = np.array([], dtype=bool).reshape(0, 0)
    mask = zivid.Mask(empty_array)

    assert mask.width == 0
    assert mask.height == 0


def test_mask_large_array():
    """Test creating Mask from large array."""
    # Test with reasonably large array
    large_array = np.random.choice([True, False], size=(100, 200))
    mask = zivid.Mask(large_array)

    assert mask.width == 200
    assert mask.height == 100

    # Verify data integrity
    result_array = mask.to_array()
    expected = large_array.astype(np.uint8)

    np.testing.assert_array_equal(result_array, expected)


def test_mask_boolean_conversion_edge_cases():
    """Test edge cases in boolean conversion."""
    # Test various falsy and truthy values
    test_cases = [
        (np.array([[True, False]]), np.array([[1, 0]], dtype=np.uint8)),
        (np.array([[1, 0]]), np.array([[1, 0]], dtype=np.uint8)),
        (np.array([[1.0, 0.0]]), np.array([[1, 0]], dtype=np.uint8)),
        (np.array([[-1, 2]]), np.array([[1, 1]], dtype=np.uint8)),
    ]

    for input_array, expected in test_cases:
        mask = zivid.Mask(input_array)
        result = mask.to_array()
        np.testing.assert_array_equal(result, expected)


def test_mask_roundtrip():
    """Test roundtrip: numpy -> Mask -> numpy."""
    original = np.random.choice([0, 1], size=(50, 75)).astype(np.uint8)

    # numpy -> Mask -> numpy
    mask = zivid.Mask(original)
    roundtrip = mask.to_array()

    np.testing.assert_array_equal(original, roundtrip)
    assert original.dtype == roundtrip.dtype
    assert original.shape == roundtrip.shape


def test_mask_internal_access():
    """Test that internal implementation is accessible."""
    mask_data = np.ones((5, 5), dtype=bool)
    mask = zivid.Mask(mask_data)

    internal_impl = zivid.mask._to_internal_mask(mask)  # pylint: disable=protected-access

    # Should be the underlying C++ object
    assert internal_impl is not None
    # The exact type depends on the C++ binding, but it should exist
    assert hasattr(internal_impl, "width") or hasattr(internal_impl, "to_string")

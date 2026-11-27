"""Tests for zivid.VectorXY class."""

import math

import pytest
import zivid


def test_vector_xy_init():
    vector = zivid.VectorXY(x=1.5, y=-2.5)

    assert vector.x == 1.5
    assert vector.y == -2.5


def test_vector_xy_setters():
    vector = zivid.VectorXY(x=1.0, y=2.0)

    vector.x = 3.0
    vector.y = 4.0

    assert vector.x == 3.0
    assert vector.y == 4.0


@pytest.mark.parametrize(
    "x, y, expected_degrees",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 90.0),
        (-1.0, 0.0, 180.0),
        (0.0, -1.0, 270.0),
        (1.0, 1.0, 45.0),
    ],
)
def test_vector_xy_angle_in_degrees(x, y, expected_degrees):
    assert zivid.VectorXY(x=x, y=y).angle_in_degrees() == pytest.approx(expected_degrees)


@pytest.mark.parametrize(
    "x, y",
    [
        (0.0, 0.0),
        (math.nan, 1.0),
        (1.0, math.inf),
    ],
)
def test_vector_xy_angle_in_degrees_raises_for_zero_length_or_non_finite(x, y):
    with pytest.raises(RuntimeError):
        zivid.VectorXY(x=x, y=y).angle_in_degrees()


@pytest.mark.parametrize(
    "x, y, expected_radians",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, math.pi / 2),
        (-1.0, 0.0, math.pi),
        (0.0, -1.0, 3 * math.pi / 2),
        (1.0, 1.0, math.pi / 4),
    ],
)
def test_vector_xy_angle_in_radians(x, y, expected_radians):
    assert zivid.VectorXY(x=x, y=y).angle_in_radians() == pytest.approx(expected_radians)


@pytest.mark.parametrize(
    "x, y",
    [
        (0.0, 0.0),
        (math.nan, 1.0),
        (1.0, math.inf),
    ],
)
def test_vector_xy_angle_in_radians_raises_for_zero_length_or_non_finite(x, y):
    with pytest.raises(RuntimeError):
        zivid.VectorXY(x=x, y=y).angle_in_radians()


def test_vector_xy_equality():
    assert zivid.VectorXY(x=1.0, y=0.0) == zivid.VectorXY(x=1.0, y=0.0)
    assert zivid.VectorXY(x=1.0, y=0.0) != zivid.VectorXY(x=0.0, y=1.0)


def test_vector_xy_str():
    assert str(zivid.VectorXY(x=12.0, y=34.0)) == "{ x: 12, y: 34 }"

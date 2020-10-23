"""Module containing the Pose class.

This module should not be imported directly by end-user, but rather accessed through
the zivid.calibration module.
"""

import _zivid


class Pose:
    """Class representing a robot pose."""

    def __init__(self, transformation_matrix):
        """Construct a Pose object.

        Args:
            transformation_matrix:  A 4x4 array representing the pose
        """

        self.__impl = _zivid.calibration.Pose(transformation_matrix)

    def to_matrix(self):
        """Get the matrix representation of the pose.

        Returns:
            A 4x4 transformation matrix
        """
        return self.__impl.to_matrix()

    def __str__(self):
        return str(self.__impl)

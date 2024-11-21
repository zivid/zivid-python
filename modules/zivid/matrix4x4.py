"""Contains Matrix4x4 class."""

import pathlib
import _zivid


class Matrix4x4(_zivid.Matrix4x4):
    """Matrix of size 4x4 containing 32-bit floats."""

    def __init__(self, arg=None):
        """
        Overloaded constructor.

        Does different kinds of initializations depending on the argument:

        * None or no arguments -> Zero initializes all values.
        * 1 dimensional List or numpy.ndarray of 16 floats -> Map 1D array of 16 values into this 2D array of 4x4.
        * 2 dimensional List or numpy.ndarray of 4x4 floats -> Copy 2D array of size 4x4.
        * str or pathlib.Path -> Load the matrix from a file.

        Args:
            arg: Any of the above-mentioned.
        """
        if arg is None:
            super().__init__()
        elif isinstance(arg, pathlib.Path):
            super().__init__(str(arg))
        else:
            super().__init__(arg)

    def inverse(self):
        """
        Return the inverse of this matrix.

        An exception is thrown if the matrix is not invertible.

        Returns:
            A new matrix, holding the inverse
        """
        return Matrix4x4(super().inverse())

    def load(self, file_path):
        """
        Load the matrix from the given file.

        Args:
            file_path: path for the file to load the matrix from.
        """
        super().load(str(file_path))

    def save(self, file_path):
        """
        Save the matrix to the given file.

        Args:
            file_path: path for the new file to save the matrix in.
        """
        super().save(str(file_path))

    @staticmethod
    def identity():
        """
        Return the identity matrix.

        Returns:
            A new matrix, holding the identity matrix.
        """
        return Matrix4x4(_zivid.Matrix4x4.identity())

    def __getitem__(self, indexes):
        """
        Access specified element with bounds checking.

        Args:
            indexes: a tuple of 2 integers as the indexes.

        Returns:
            The accessed item.
        """
        return super()._getitem(indexes)

    def __iter__(self):
        """
        Return an iterator to iterate all 16 elements as if this is a 1D array.

        Returns:
            the iterator.
        """
        iterator = super().__iter__()
        return iterator

    def __setitem__(self, indexes, value):
        """
        Set specified element with bounds checking.

        Args:
            indexes: a tuple of 2 integers as the indexes.
            value: value to set the specific item to.
        """
        super()._setitem(indexes, value)

"""Contains the Matrix class."""
from pathlib import Path
import numpy as np
import yaml


class Matrix:
    """A floating point matrix.

    It is a simple wrapper around a NumPy matrix, with the capability 
    to save and load from Zivid's YAML file format.
    """

    def __init__(self, file_name):
        """Create a matrix by loading it from a YAML file.

        Args:
            file_name: A pathlib.Path instance or a string specifying a YAML file

        Raises:
            TypeError: Unsupported type provided for file name
        """
        if isinstance(file_name, (str, Path)):
            self.load(file_name)
        else:
            raise TypeError(
                "Unsupported type for argument file_name. Got {}, expected {} or {}.".format(
                    type(file_name), str.__name__, Path.__name__
                )
            )

    def __str__(self):
        return str(self.__impl)

    def as_numpy_array(self):
        """Get the matrix as a NumPy array.

        Returns:
            The NumPy array
        """
        return self.__impl

    def save(self, file_path):
        """Save the matrix to a YAML file.

        Args:
            file_path: A pathlib.Path instance or a string specifying the destination
        """
        with open(file_path,"w") as f:
            yaml.safe_dump({
                "__version__": {
                    "serializer": 1,
                    "data": 1
                }
            }, f, sort_keys=False)
            yaml.safe_dump({
                "FloatMatrix": {
                    "Data": self.as_numpy_array().tolist()
                }
            }, f, default_flow_style=None)

    def load(self, file_path):
        """Load the matrix from a Zivid YAML file.

        Args:
            file_path: A pathlib.Path instance or a string specifying a YAML file to load
        """
        with open(file_path) as f:
            self.__impl = np.array(yaml.safe_load(f)["FloatMatrix"]["Data"])

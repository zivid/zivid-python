"""Contains a the Image class."""
import numpy

import _zivid


class Image:
    """A 2-dimensional image stored on the host."""

    def __init__(self, internal_image):
        """Can only be initialized with an zivid internal image.

        Args:
            internal_image: an internal image

        Raises:
            TypeError: unsupported type provided for internal image

        """
        if not isinstance(internal_image, _zivid.Image):
            raise TypeError(
                "Unsupported type for argument internal_image. Got {}, expected {}".format(
                    type(internal_image), type(_zivid.Image)
                )
            )
        self.__impl = internal_image

    def save(self, file_path):
        """Save the image to a file.

        The supported file type is PNG with extension .png.
        This method will throw an exception if failing to save to the provided file_path.

        Args:
            file_path: destination path

        """
        self.__impl.save(str(file_path))

    def to_array(self):
        """Convert the image into a numpy array.

        Returns:
            a numpy array

        """
        self.__impl.assert_not_released()
        return numpy.array(self.__impl)

    def release(self):
        """Release the underlying resources."""
        self.__impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()

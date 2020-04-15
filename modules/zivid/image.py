"""Contains a the Image class."""
import numpy

import _zivid


class Image:
    """A two-dimensional image stored on the host."""

    def __init__(self, impl):
        """Initialize Image wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.ImageRGBA):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.ImageRGBA)
                )
            )
        self.__impl = impl

    @property
    def height(self):
        """Get the height of the image (number of rows).

        Returns:
            A positive integer
        """
        return self.__impl.height()

    @property
    def width(self):
        """Get the width of the image (number of columns).

        Returns:
            A positive integer
        """
        return self.__impl.width()

    def save(self, file_path):
        """Save the image to a file.

        The supported file type is PNG with extension .png.
        This method will throw an exception if failing to save to the provided file_path.

        Args:
            file_path: A pathlib.Path instance or a string specifying destination
        """
        self.__impl.save(str(file_path))

    def copy_data(self):
        """Copy image data to numpy array.

        Returns:
            A numpy array containing color pixel data
        """
        self.__impl.assert_not_released()
        return numpy.array(self.__impl)

    def release(self):
        """Release the underlying resources."""
        try:
            impl = self.__impl
        except AttributeError:
            pass
        else:
            impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()

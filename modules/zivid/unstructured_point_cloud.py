"""Contains the UnstructuredPointCloud class."""

import numpy

import _zivid


class UnstructuredPointCloud:

    def __init__(self, impl):
        """Initialize UnstructuredPointCloud wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.UnstructuredPointCloud):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.UnstructuredPointCloud
                )
            )
        self.__impl = impl

    def copy_data(self, data_format):

        self.__impl.assert_not_released()

        data_formats = {
            "xyz": _zivid.UnstructuredPointCloud.copy_points_xyz,
            "rgba": _zivid.UnstructuredPointCloud.copy_colors_rgba,
            "snr": _zivid.UnstructuredPointCloud.copy_snrs,
        }

        try:
            data_copy_function = data_formats[data_format]
        except KeyError as ex:
            raise ValueError(
                "Unsupported data format: {data_format}. Supported formats: {all_formats}".format(
                    data_format=data_format, all_formats=list(data_formats.keys())
                )
            ) from ex
        return data_copy_function(self.__impl)

    @property
    def size(self):
        return self.__impl.size()

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

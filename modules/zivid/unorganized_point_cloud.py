"""Contains the UnorganizedPointCloud class."""

import numpy

import _zivid


class UnorganizedPointCloud:

    def __init__(self, impl):
        """Initialize UnorganizedPointCloud wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.UnorganizedPointCloud):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.UnorganizedPointCloud
                )
            )
        self.__impl = impl

    def extended(self, other):
        """
        TODO(ESKIL): Docstring
        """
        if not isinstance(other, UnorganizedPointCloud):
            raise TypeError(
                "Unsupported type for argument other. Got {}, expected {}".format(type(other), UnorganizedPointCloud)
            )
        return UnorganizedPointCloud(self.__impl.extended(other.__impl))

    def voxel_downsampled(self, voxel_size, min_points_per_voxel):
        """
        TODO(ESKIL): Docstring
        """
        return UnorganizedPointCloud(self.__impl.voxel_downsampled(voxel_size, min_points_per_voxel))

    def transform(self, matrix):
        """Transform the point cloud in-place by a 4x4 transformation matrix.

        The transform matrix must be affine, i.e., the last row of the matrix should be [0, 0, 0, 1].

        Args:
            matrix: A 4x4 numpy arrays of floats

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        self.__impl.transform(matrix)
        return self

    def copy_data(self, data_format):

        self.__impl.assert_not_released()

        data_formats = {
            "xyz": _zivid.UnorganizedPointCloud.copy_points_xyz,
            "rgba": _zivid.UnorganizedPointCloud.copy_colors_rgba,
            "bgra": _zivid.UnorganizedPointCloud.copy_colors_bgra,
            "snr": _zivid.UnorganizedPointCloud.copy_snrs,
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
        """
        TODO(ESKIL): Docstring
        """
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

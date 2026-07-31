"""Contains the PointCloud class."""

from __future__ import annotations

import _zivid
import numpy
from zivid.device_array import DeviceArray, _require_stream_or_queue
from zivid.image import Image
from zivid.mask import Mask, _to_internal_mask
from zivid.pixel_format import PixelFormat, _resolve_color_format
from zivid.settings import _to_internal_settings_region_of_interest_box
from zivid.unorganized_point_cloud import UnorganizedPointCloud

_COLOR_FORMAT_ACCESSOR_SUFFIX = {
    PixelFormat.RGBA: "rgba",
    PixelFormat.RGBA_SRGB: "rgba_srgb",
    PixelFormat.BGRA: "bgra",
    PixelFormat.BGRA_SRGB: "bgra_srgb",
    PixelFormat.RGBAF: "rgbaf",
}


class PointCloud:
    """Point cloud with x, y, z, RGB and color laid out on a 2D grid.

    An instance of this class is a handle to a point cloud stored on the compute device memory.
    Use the method copy_data to copy point cloud data from the compute device to a numpy
    array in host memory. Several formats are available.

    If the point cloud is the result of a 2D+3D capture, the RGB colors will be set from the captured 2D color image.
    If different pixel sampling (resolution) settings for 2D and 3D were used, or if the point cloud is upsampled or
    downsampled, then the RGB colors will be resampled to correspond 1:1 with the 3D point cloud resolution.
    To get the original resolution 2D color image from the 2D+3D capture, see the frame_2d method of the Frame class.

    If the point cloud is the result of a 3D-only capture, the RGB colors will be set to a uniform default color.
    """

    class Downsampling:  # pylint: disable=too-few-public-methods
        """Collection of valid options to PointCloud.downsample()."""

        by2x2 = "by2x2"
        by3x3 = "by3x3"
        by4x4 = "by4x4"

        _valid_values = {
            "by2x2": _zivid.PointCloud.Downsampling.by2x2,
            "by3x3": _zivid.PointCloud.Downsampling.by3x3,
            "by4x4": _zivid.PointCloud.Downsampling.by4x4,
        }

        @classmethod
        def valid_values(cls) -> list:
            """Get list of allowed values.

            Returns:
                List of strings
            """
            return list(cls._valid_values.keys())

    def __init__(self, impl):
        """Initialize PointCloud wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.PointCloud):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(type(impl), _zivid.PointCloud)
            )
        self.__impl = impl

    def copy_data(self, data_format: str) -> numpy.ndarray:
        """Copy point cloud data from GPU to numpy array.

        Supported data formats:
        xyz:            ndarray(Height,Width,3) of float
        xyzw:           ndarray(Height,Width,4) of float
        z:              ndarray(Height,Width)   of float
        rgba:           ndarray(Height,Width,4) of uint8
        bgra:           ndarray(Height,Width,4) of uint8
        rgba_srgb:      ndarray(Height,Width,4) of uint8
        bgra_srgb:      ndarray(Height,Width,4) of uint8
        srgb:           ndarray(Height,Width,4) of uint8 (deprecated, use rgba_srgb instead)
        normals:        ndarray(Height,Width,3) of float
        snr:            ndarray(Height,Width)   of float
        xyzrgba:        ndarray(Height,Width)   of composite dtype (accessed with e.g. arr["x"])
        xyzbgra:        ndarray(Height,Width)   of composite dtype (accessed with e.g. arr["x"])
        xyzrgba_srgb:   ndarray(Height,Width)   of composite dtype (accessed with e.g. arr["x"])
        xyzbgra_srgb:   ndarray(Height,Width)   of composite dtype (accessed with e.g. arr["x"])

        Args:
            data_format: A string specifying the data to be copied

        Returns:
            A numpy array with the requested data.

        Raises:
            ValueError: if the requested data format does not exist
        """
        self.__impl.assert_not_released()

        data_formats = {
            "xyz": _zivid.Array2DPointXYZ,
            "xyzw": _zivid.Array2DPointXYZW,
            "z": _zivid.Array2DPointZ,
            "rgba": _zivid.Array2DColorRGBA,
            "bgra": _zivid.Array2DColorBGRA,
            "rgba_srgb": _zivid.Array2DColorRGBA_SRGB,
            "bgra_srgb": _zivid.Array2DColorBGRA_SRGB,
            "srgb": _zivid.Array2DColorRGBA_SRGB,
            "normals": _zivid.Array2DNormalXYZ,
            "snr": _zivid.Array2DSNR,
            "xyzrgba": _zivid.Array2DPointXYZColorRGBA,
            "xyzbgra": _zivid.Array2DPointXYZColorBGRA,
            "xyzrgba_srgb": _zivid.Array2DPointXYZColorRGBA_SRGB,
            "xyzbgra_srgb": _zivid.Array2DPointXYZColorBGRA_SRGB,
        }
        try:
            data_format_class = data_formats[data_format]
        except KeyError as ex:
            raise ValueError(
                "Unsupported data format: {data_format}. Supported formats: {all_formats}".format(
                    data_format=data_format, all_formats=list(data_formats.keys())
                )
            ) from ex
        return numpy.array(data_format_class(self.__impl))

    def copy_image(self, data_format: str) -> Image:
        """Copy the point cloud colors as 8-bit image in input format.

        Supported data formats:
        rgba:       Image(Height,Width,4) of uint8
        bgra:       Image(Height,Width,4) of uint8
        rgba_srgb:  Image(Height,Width,4) of uint8
        bgra_srgb:  Image(Height,Width,4) of uint8
        srgb:       Image(Height,Width,4) of uint8 (deprecated, use rgba_srgb instead)

        Args:
            data_format: A string specifying the image data format

        Returns:
            An image instance containing color data

        Raises:
            ValueError: if the requested data format does not exist
        """
        self.__impl.assert_not_released()

        supported_color_formats = ["rgba", "bgra", "rgba_srgb", "bgra_srgb", "srgb"]

        if data_format == "rgba":
            return Image(self.__impl.copy_image_rgba())
        if data_format == "bgra":
            return Image(self.__impl.copy_image_bgra())
        if data_format in ("rgba_srgb", "srgb"):
            return Image(self.__impl.copy_image_rgba_srgb())
        if data_format == "bgra_srgb":
            return Image(self.__impl.copy_image_bgra_srgb())
        raise ValueError(
            "Unsupported color format: {data_format}. Supported formats: {all_formats}".format(
                data_format=data_format, all_formats=supported_color_formats
            )
        )

    def clone(self) -> PointCloud:
        """Get a clone of the point cloud.

        The clone will include a copy of all the point cloud data on the compute device memory. This means that the
        returned point cloud will not be affected by subsequent modifications (such as transform or downsample) on the
        original point cloud.

        This function incurs a performance cost due to the copying of the compute device memory. When performance is
        important we recommend to avoid using this method, and instead modify the existing point cloud.

        This method is equivalent to calling `copy.deepcopy` on the point cloud. You can obtain a shallow copy that does
        not copy the underlying data by using `copy.copy` on the point cloud instead.

        Returns:
            A new PointCloud instance
        """
        return PointCloud(self.__impl.clone())

    def transform(self, matrix: numpy.ndarray) -> PointCloud:
        """Transform the point cloud in-place by a 4x4 transformation matrix.

        The transform matrix must be affine, i.e., the last row of the matrix should be [0, 0, 0, 1].

        Args:
            matrix: A 4x4 numpy arrays of floats

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        self.__impl.transform(matrix)
        return self

    def transformed(self, matrix: numpy.ndarray) -> PointCloud:
        """Get a transformed copy of the point cloud.

        This method is identical to "transform", except the transformed point cloud is
        returned as a new PointCloud instance. The current point cloud is not modified.

        Args:
            matrix: A 4x4 numpy arrays of floats

        Returns:
            A new PointCloud instance
        """
        return PointCloud(self.__impl.transformed(matrix))

    @property
    def transformation_matrix(self) -> numpy.ndarray:
        """Return the current transformation matrix of this point cloud.

        Returns the transformation matrix from the camera's native coordinate system to the current
        coordinate system. The returned matrix represents the cumulative result of all the transform operations
        performed on the point cloud. If no transformations have been applied, the identity matrix is returned.

        Note: ZDF files saved from SDK 2.14 or earlier did not store the active transformation matrix. This
        means that for point clouds loaded from .zdf files from SDK 2.14 or earlier, this method will always
        return the identity matrix even in the case where the point cloud had been transformed prior to saving.

        Returns:
            A 4x4 numpy array of floats
        """
        return self.__impl.transformation_matrix()

    def downsample(self, downsampling: str) -> PointCloud:
        """Downsample the point cloud in-place.

        Downsampling is used to reduce the number of points in the point cloud. Downsampling is performed
        by combining a 2x2, 3x3 or 4x4 region of pixels in the original point cloud to one pixel in the
        new point cloud. A downsampling factor of 2x2 will reduce width and height each to half, and thus
        the overall number of points to 1/4. 3x3 downsampling reduces width and height each to 1/3, and
        the overall number of points to 1/9, and so on.

        X, Y and Z coordinates are downsampled by computing the SNR^2 weighted average of each point in
        the corresponding NxN region in the original point cloud, ignoring invalid (NaN) points. Color is
        downsampled by computing the average value for each color channel in the NxN region. SNR value is
        downsampled by computing the square root of the sum of SNR^2 of each valid (non-NaN) point in the
        NxN region. If all points in the NxN region are invalid (NaN), the downsampled SNR is set to the
        max SNR in the region.

        As an alternative to using this method, downsampling may also be specified up-front when capturing
        by using Settings/Processing/Resampling.

        Downsampling is performed on the compute device. The point cloud is modified in-place. Use
        "downsampled" if you want to downsample to a new PointCloud instance. Downsampling
        can be repeated multiple times to further reduce the size of the point cloud, if desired.

        Note that the width or height of the point cloud is not required to divide evenly by the
        downsampling factor (2, 3 or 4). The new width and height equals the original width and height
        divided by the downsampling factor, rounded down. In this case the remaining columns at the right
        and/or rows at the bottom of the original point cloud are ignored.

        Args:
            downsampling: One of the strings in PointCloud.Downsample.valid_values()

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        internal_downsampling = PointCloud.Downsampling._valid_values[downsampling]  # pylint: disable=protected-access
        self.__impl.downsample(internal_downsampling)
        return self

    def downsampled(self, downsampling: str) -> PointCloud:
        """Get a downsampled copy of the point cloud.

        This method is identical to "downsample", except the downsampled point cloud is
        returned as a new PointCloud instance. The current point cloud is not modified.

        Args:
            downsampling: One of the strings in PointCloud.Downsample.valid_values()

        Returns:
            A new PointCloud instance
        """
        internal_downsampling = PointCloud.Downsampling._valid_values[downsampling]  # pylint: disable=protected-access
        return PointCloud(self.__impl.downsampled(internal_downsampling))

    def mask_by_region_of_interest(self, roi_box) -> PointCloud:
        """Apply a region of interest box mask to the point cloud in-place.

        Region of interest masking is used to mask out points that fall outside a specified 3D box region.
        Points outside the region are set to invalid (NaN) values, effectively removing them from the point cloud
        while maintaining the original dimensions and structure.

        The masking is performed on the compute device. The point cloud is modified in-place. Use
        "masked_by_region_of_interest" if you want to apply ROI masking to a new PointCloud instance.

        The ROI box must be enabled (roi_box.enabled == True) for the masking to be applied.

        Args:
            roi_box: A zivid.Settings.RegionOfInterest.Box instance defining the 3D region to preserve

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        internal_roi_box = _to_internal_settings_region_of_interest_box(roi_box)
        self.__impl.mask_by_region_of_interest(internal_roi_box)
        return self

    def masked_by_region_of_interest(self, roi_box) -> PointCloud:
        """Apply region of interest filtering to a copy of the point cloud.

        This method is identical to "mask_by_region_of_interest", except that the filtering is
        performed on a copy of the original point cloud. This method does not modify the original
        point cloud.

        Args:
            roi_box: A zivid.Settings.RegionOfInterest.Box instance defining the 3D region to preserve

        Returns:
            A new PointCloud instance
        """
        internal_roi_box = _to_internal_settings_region_of_interest_box(roi_box)
        return PointCloud(self.__impl.masked_by_region_of_interest(internal_roi_box))

    def mask(self, mask: numpy.ndarray | Mask) -> PointCloud:
        """Apply a binary mask to the point cloud in-place.

        The mask indicates which points in the point cloud should be considered invalid (NaN).
        The mask can be a 2D numpy array of booleans/uint8 or a zivid.Mask object with the same
        height and width as the point cloud. A value of True/non-zero in the mask indicates that
        the corresponding point in the point cloud should be set to invalid (NaN). A value of
        False/zero indicates that the corresponding point should be kept unchanged.

        Args:
            mask: A binary mask as a 2D numpy array of booleans/uint8 or a zivid.Mask object

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        if isinstance(mask, Mask):
            # Already a Mask object, use its internal implementation
            self.__impl.mask(_to_internal_mask(mask))
        else:
            # Convert numpy array or other data to Mask first
            mask_obj = Mask(mask)
            self.__impl.mask(_to_internal_mask(mask_obj))
        return self

    def masked(self, mask: numpy.ndarray | Mask) -> PointCloud:
        """Get a copy of the point cloud with a binary mask applied.

        This method is identical to "mask", except the masked point cloud is
        returned as a new PointCloud instance. The current point cloud is not modified.

        Args:
            mask: A binary mask as a 2D numpy array of booleans/uint8 or a zivid.Mask object

        Returns:
            A new PointCloud instance with the mask applied
        """
        if isinstance(mask, Mask):
            # Already a Mask object, use its internal implementation
            return PointCloud(self.__impl.masked(_to_internal_mask(mask)))
        # Convert numpy array or other data to Mask first
        mask_obj = Mask(mask)
        return PointCloud(self.__impl.masked(_to_internal_mask(mask_obj)))

    @property
    def height(self) -> int:
        """Get the height of the point cloud (number of rows).

        Returns:
            A positive integer
        """
        return self.__impl.height()

    @property
    def width(self) -> int:
        """Get the width of the point cloud (number of columns).

        Returns:
            A positive integer
        """
        return self.__impl.width()

    def to_unorganized_point_cloud(self) -> UnorganizedPointCloud:
        """Convert to an UnorganizedPointCloud.

        The PointCloud class represents an organized point cloud, meaning that it contains 3D data
        for every pixel (row, col) on the sensor. If 3D data could not be computed for a given pixel,
        or was removed by a filter, that pixel contains (x,y,z) = (NaN, NaN, NaN) representing an
        "invalid" point. For some use cases it is more useful to have an unorganized point cloud,
        which instead contains a linear list of only valid points.

        This function efficiently picks out the valid (not-NaN) points from this structured point cloud,
        and constructs an unorganized point cloud containing the XYZ, Color and SNR from those points only.
        The resulting unorganized point cloud will have less than (or equal) memory footprint compared to
        the structured point cloud it comes from, since it will contain fewer (or equal) number of points.

        Returns:
            A new UnorganizedPointCloud constructed from the data in this PointCloud
        """
        return UnorganizedPointCloud(self.__impl.to_unorganized_point_cloud())

    def device_points_xyz(self, stream_or_queue) -> DeviceArray:
        """Get a GPU device array containing XYZ point coordinates.

        Returns a DeviceArray providing access to point cloud data
        on the GPU device without CPU transfers.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr to synchronize SDK
                operations with before handing off the buffer.

        Returns:
            A DeviceArray object containing XYZ point coordinates.
        """
        self.__impl.assert_not_released()
        _require_stream_or_queue(stream_or_queue)
        return DeviceArray(self.__impl.device_points_xyz(stream_or_queue))

    def device_points_xyzw(self, stream_or_queue) -> DeviceArray:
        """Get a GPU device array containing XYZW point coordinates.

        Returns a DeviceArray providing access to point cloud data
        on the GPU device without CPU transfers.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr to synchronize SDK
                operations with before handing off the buffer.

        Returns:
            A DeviceArray object containing XYZW point coordinates.
        """
        self.__impl.assert_not_released()
        _require_stream_or_queue(stream_or_queue)
        return DeviceArray(self.__impl.device_points_xyzw(stream_or_queue))

    def device_points_z(self, stream_or_queue) -> DeviceArray:
        """Get a GPU device array containing Z point coordinates.

        Returns a DeviceArray providing access to point cloud data
        on the GPU device without CPU transfers.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr to synchronize SDK
                operations with before handing off the buffer.

        Returns:
            A DeviceArray object containing Z point coordinates.
        """
        self.__impl.assert_not_released()
        _require_stream_or_queue(stream_or_queue)
        return DeviceArray(self.__impl.device_points_z(stream_or_queue))

    def device_snrs(self, stream_or_queue) -> DeviceArray:
        """Get a GPU device array containing SNR values.

        Returns a DeviceArray providing access to point cloud data
        on the GPU device without CPU transfers.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr to synchronize SDK
                operations with before handing off the buffer.

        Returns:
            A DeviceArray object containing SNR values.
        """
        self.__impl.assert_not_released()
        _require_stream_or_queue(stream_or_queue)
        return DeviceArray(self.__impl.device_snrs(stream_or_queue))

    def device_normals_xyz(self, stream_or_queue) -> DeviceArray:
        """Get a GPU device array containing normal vectors.

        Returns a DeviceArray providing access to point cloud data
        on the GPU device without CPU transfers.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr to synchronize SDK
                operations with before handing off the buffer.

        Returns:
            A DeviceArray object containing normal vectors.
        """
        self.__impl.assert_not_released()
        _require_stream_or_queue(stream_or_queue)
        return DeviceArray(self.__impl.device_normals_xyz(stream_or_queue))

    def device_image(self, stream_or_queue, color_format: PixelFormat) -> DeviceArray:
        """Get a GPU device array containing the organized color image.

        Returns a DeviceArray providing access to point cloud color data on the GPU device without
        CPU transfers.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr the SDK records a readiness
                event on before handing off the buffer.
            color_format: A zivid.PixelFormat color format. Supported: RGBA, BGRA, RGBA_SRGB,
                BGRA_SRGB, RGBAF.

        Returns:
            A DeviceArray object containing the color image data.
        """
        suffix = _resolve_color_format(color_format, _COLOR_FORMAT_ACCESSOR_SUFFIX)
        self.__impl.assert_not_released()
        _require_stream_or_queue(stream_or_queue)
        accessor = getattr(self.__impl, "device_image_{}".format(suffix))
        return DeviceArray(accessor(stream_or_queue))

    def release(self) -> None:
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

    def __copy__(self):
        return PointCloud(self.__impl.__copy__())

    def __deepcopy__(self, memodict):
        return PointCloud(self.__impl.__deepcopy__(memodict))

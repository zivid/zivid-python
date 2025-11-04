"""Module for experimental projection features. This API may change in the future."""

import _zivid
from zivid.frame_2d import Frame2D
from zivid.settings import Settings, _to_internal_settings
from zivid.settings2d import Settings2D, _to_internal_settings2d


class ProjectedImage:
    """A handle to a 2D image being displayed on a Zivid camera's projector.

    The image projection will stop either when the instance is destroyed, when the stop() method is called,
    or when exiting the "with" block if used as a context manager.
    """

    def __init__(self, impl):
        """Initialize ProjectedImage wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl: Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if isinstance(impl, _zivid.ProjectedImage):
            self.__impl = impl
        else:
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}.".format(type(impl), _zivid.ProjectedImage)
            )

    def __str__(self):
        return str(self.__impl)

    def capture_2d(self, settings):
        """Capture a single 2D frame without stopping the ongoing image projection.

        This method returns right after the acquisition of the image is complete.

        To match the projector frequency of 120 Hz, the exposure time should be set to 8333 microseconds,
        otherwise the projected image brightness in the captured image may appear inconsistent over repeated
        captures.

        Some settings cannot be used to capture while projecting, depending on the camera model:
            * Not all camera models support capturing rgb colors while projecting. An exception will be thrown
              if `Settings2D/Sampling/Color` is set to `rgb` and the camera does not support capturing rgb
              colors while projecting. Those cameras must use `grayscale` instead.

            * When `Settings2D/Sampling/Color` is set to `rgb` the brightness must be 0 in all acquisitions,
              otherwise an exception will be thrown.

            * When `Settings2D/Sampling/Color` is set to `grayscale` the brightness may be set to some
              value > 0. This value will override the brightness of the projection to control the contrast
              between the scene and the projected image. This will cause the projected image to flicker while
              the camera is capturing. If the brightness is set to 0 the image projection will be
              uninterrupted.

        Args:
            settings: Settings to use for the capture. Can be either a Settings2D instance or a Settings instance.
                      If a Settings instance is provided, only the Settings.color part is used. An exception is thrown
                      if the Settings.color part is not set.

        Returns:
            A Frame2D containing a 2D image plus metadata.

        Raises:
            TypeError: If argument is not either a Settings2D or a Settings instance.
        """
        if isinstance(settings, Settings2D):
            return Frame2D(self.__impl.capture_2d(_to_internal_settings2d(settings)))
        if isinstance(settings, Settings):
            return Frame2D(self.__impl.capture_2d(_to_internal_settings(settings)))
        raise TypeError(
            "Unsupported settings type, expected: {expected_types}, got: {value_type}".format(
                expected_types=" or ".join([Settings.__name__, Settings2D.__name__]),
                value_type=type(settings),
            )
        )

    def capture(self, settings2d):
        """Capture a single 2D frame without stopping the ongoing image projection.

        This method is deprecated as of SDK 2.15 and will be removed in the next SDK major version (3.0).
        Use capture_2d instead.

        This method returns right after the acquisition of the image is complete.

        To match the projector frequency of 120 Hz, the exposure time should be set to 8333 microseconds,
        otherwise the projected image brightness in the captured image may appear inconsistent over repeated
        captures.

        Some settings cannot be used to capture while projecting, depending on the camera model:
            * Not all camera models support capturing rgb colors while projecting. An exception will be thrown
              if `Settings2D/Sampling/Color` is set to `rgb` and the camera does not support capturing rgb
              colors while projecting. Those cameras must use `grayscale` instead.

            * When `Settings2D/Sampling/Color` is set to `rgb` the brightness must be 0 in all acquisitions,
              otherwise an exception will be thrown.

            * When `Settings2D/Sampling/Color` is set to `grayscale` the brightness may be set to some
              value > 0. This value will override the brightness of the projection to control the contrast
              between the scene and the projected image. This will cause the projected image to flicker while
              the camera is capturing. If the brightness is set to 0 the image projection will be
              uninterrupted.

        Args:
            settings2d: A Settings2D instance to be used for 2D capture.

        Returns:
            A Frame2D containing a 2D image plus metadata.

        Raises:
            TypeError: If argument is not a Settings2D.
        """
        if isinstance(settings2d, Settings2D):
            return Frame2D(self.__impl.capture(_to_internal_settings2d(settings2d)))
        raise TypeError("Unsupported settings type: {}".format(type(settings2d)))

    def stop(self):
        """Stop the ongoing image projection."""
        self.__impl.stop()

    def active(self):
        """Check if a handle is associated with an ongoing image projection.

        Returns:
            A boolean indicating projection state.
        """
        return self.__impl.active()

    def release(self):
        """Release the underlying resources and stop projection."""
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


def projector_resolution(camera):
    """Get the resolution of the internal projector in the Zivid camera.

    Args:
        camera: The Camera instance to get the projector resolution of.

    Returns:
        The resolution as a tuple (height,width).
    """
    return _zivid.projection.projector_resolution(camera._Camera__impl)  # pylint: disable=protected-access


def show_image_bgra(camera, image_bgra):
    """Display a 2D color image using the projector.

    The image resolution needs to be the same as the resolution obtained from the projector_resolution
    function for the camera. This function returns a ProjectedImage instance. Projection will continue
    until this object is destroyed, if its stop() method is called, or if another capture is initialized
    on the same camera. The ProjectedImage object can be used as a context manager, in which case the
    projection will stop when exiting the "with" block.

    Args:
        camera: The Camera instance to project with.
        image_bgra: The image to project in the form of a HxWx4 numpy array with BGRA colors.

    Returns:
        A handle in the form of a ProjectedImage instance.
    """
    return ProjectedImage(
        _zivid.projection.show_image_bgra(
            camera._Camera__impl,  # pylint: disable=protected-access
            image_bgra,
        )
    )


def pixels_from_3d_points(camera, points):
    """Get 2D projector pixel coordinates corresponding to 3D points relative to the camera.

    This function takes 3D points in the camera's reference frame and converts them to the projector frame
    using the internal calibration of a Zivid camera. In a Zivid point cloud, each point corresponds to a
    pixel coordinate in the camera, but the projector has a slight offset. The translation of each point
    depends on the distance between the camera and the point, as well as the distance and angle between the
    camera and the projector.

    Args:
        camera: The Camera instance that the 3D points are in the frame of.
        points: A list of 3D (XYZ) points as List[List[float[3]]] or Nx3 Numpy array.

    Returns:
        The corresponding 2D (XY) points in the projector (List[List[float[2]]])
    """
    return _zivid.projection.pixels_from_3d_points(
        camera._Camera__impl,  # pylint: disable=protected-access
        points,
    )

"""Contains the Preset and Category classes."""
import _zivid

from zivid.settings import _to_settings
from zivid.camera_info import CameraInfo, _to_internal_camera_info


class Preset:
    """3D settings preset.

    Presets are pre-defined settings that are tuned for different camera models to perform optimally
    under different conditions and use cases.

    This class cannot be initialized directly by the end-user. Use the presets method on the
    Category class to obtain a Preset instance.
    """

    def __init__(self, impl):
        """Initialize Preset wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.presets.Preset):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.presets.Preset
                )
            )
        self.__impl = impl

    @property
    def name(self):
        """Get the name of the preset.

        Returns:
            The name of the preset.
        """
        return self.__impl.name()

    @property
    def settings(self):
        """Get the settings of the preset.

        The settings might change between different releases of the SDK. New presets might be added
        and old ones might be removed. If having the exact same settings is desired, it is
        recommended to save them to a YML file and load as needed.

        Returns:
                The settings of the preset.
        """
        return _to_settings(self.__impl.settings())

    def __str__(self):
        return str(self.__impl)


class Category:
    """Preset category.

    This class cannot be initialized directly by the end-user. Use the categories function to
    obtain a list of categories available for a camera.
    """

    def __init__(self, impl):
        """Initialize Category wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.presets.Category):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.presets.Category
                )
            )
        self.__impl = impl

    @property
    def name(self):
        """Get the name of the category.

        Returns:
            The name of the category.
        """
        return self.__impl.name()

    @property
    def presets(self):
        """Get the presets available in the category.

        The settings might change between different releases of the SDK. New presets might be added
        and old ones might be removed. If having the exact same settings is desired, it is
        recommended to save them to a YML file and load as needed.

        Returns:
            The presets available in the category.
        """
        return [Preset(p) for p in self.__impl.presets()]

    def __str__(self):
        return str(self.__impl)


def categories(model):
    """Get available preset categories for the specified camera model.

    A preset category contains a collection of presets optimized for one scenario or use case.

    The settings might change between different releases of the SDK. New presets might be added
    and old ones might be removed. If having the exact same settings is desired, it is recommended
    to save them to a YML file and load as needed.

    Args:
        model: The model for the camera whose preset categories should be returned. This value can
               be obtained from CameraInfo.model.

    Returns:
        The available categories for the specified camera model.
    """
    return [
        Category(c)
        for c in _zivid.presets.categories(
            _to_internal_camera_info(CameraInfo(model=model)).model
        )
    ]

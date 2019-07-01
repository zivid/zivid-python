"""Get version information for the library."""
import _zivid


class SDKVersion:  # pylint: disable=too-few-public-methods
    """Get the version of the loaded library."""

    major = _zivid.version.major
    minor = _zivid.version.minor
    patch = _zivid.version.patch
    build = _zivid.version.build
    full = _zivid.version.full

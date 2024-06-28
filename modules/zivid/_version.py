"""Query module version."""

import sys


def get_version(module_name):
    # pylint: disable=import-outside-toplevel
    """Return the version of a module.

    Args:
        module_name: Name of a module

    Returns:
        The module version or None if not found.
    """

    # pkg_resources is deprecated and throws a warning in newer Python verisons.
    # We use importlib.metadata instead, but this is only available since Python 3.8.
    if sys.version_info < (3, 8):
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            return get_distribution(module_name).version
        except DistributionNotFound:
            return None

    from importlib.metadata import version, PackageNotFoundError

    try:
        return version(module_name)
    except PackageNotFoundError:
        return None

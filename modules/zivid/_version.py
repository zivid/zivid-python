"""Query module version."""
from importlib.metadata import version, PackageNotFoundError


def get_version(module_name):
    """Return the version of a module.

    Args:
        module_name: Name of a module

    Returns:
        The module version or None if not found.
    """
    try:
        return version(module_name)
    except PackageNotFoundError:
        return None

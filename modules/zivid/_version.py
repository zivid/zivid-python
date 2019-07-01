"""Query module version."""
from pkg_resources import get_distribution, DistributionNotFound


def get_version(module_name):
    """Return the version module version.

    Args:
        module_name: Name of a module

    Returns:
        The module version

    """
    try:
        return get_distribution(module_name).version
    except DistributionNotFound:
        return None

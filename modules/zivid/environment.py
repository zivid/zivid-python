"""Zivid environment, configured through environment variables."""
from pathlib import Path
import _zivid


def data_path():
    """Get default location for sample data.

    The default location for sample data can be configured through the ZIVID_DATA environment variable

    Returns:
        The data folder

    """
    return Path(_zivid.environment.data_path())

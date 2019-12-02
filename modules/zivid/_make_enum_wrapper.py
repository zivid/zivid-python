"""Contains functions to wrapp internal into public enums."""
from enum import EnumMeta, Enum, unique


def _make_enum_wrapper(internal_enum, docstring):
    """Wrap internal enum type to public enum.

    Args:
        internal_enum: an internal enum object
        docstring: an instance of str type

    Returns:
        an enum class

    """

    class _WrappedEnumMeta(EnumMeta):
        """Scan `internal_enum` and add corresponding members to `oldclassdict`."""

        def __new__(cls, metacls, bases, oldclassdict):
            for member in internal_enum.__members__:
                oldclassdict[member] = str(member)
            return super().__new__(cls, metacls, bases, oldclassdict)

    @unique
    class WrappedEnum(str, Enum, metaclass=_WrappedEnumMeta):

        __doc__ = docstring

        def _to_internal(self):
            return getattr(self._internal_enum, self.name)

        def __str__(self):
            return str(self.name)

    WrappedEnum._internal_enum = internal_enum  # pylint: disable=protected-access
    WrappedEnum.__name__ = internal_enum.__name__  # pylint: disable=protected-access
    return WrappedEnum

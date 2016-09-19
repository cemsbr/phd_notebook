"""Parse HiBench Sort log files."""
from lib.parser import Parser


class HBSortParser:
    """Parse HiBench Sort log files."""

    #: Minimum input size of target set.
    THRESHOLD = 3284983900

    @classmethod
    def get_set(cls, size):
        """Whether the app belongs to the profiling or target set.

        Args:
            size (int): Input size in bytes.

        Returns:
            str: 'profiling' or 'target'.
        """
        return 'profiling' if size < cls.THRESHOLD else 'target'

    @staticmethod
    def get_apps():
        """Return a generator for all apps."""
        parser = Parser()
        apps = parser.get_apps('hibench', 'sort', '*')
        return (app for app in apps if len(app.slaves) not in [12, 123])

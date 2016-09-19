"""Common code for parsing Apache Spark log files."""
from os import path

from sparklogstats import LogParser as LogParser


class Parser:
    """Tell whether read method is only from memory.

    Accumulate all different read methods found.
    """

    #: Read methods considered to be from memory.
    MEM_METHODS = set([None, 'Memory'])

    @classmethod
    def fits_in_memory(cls, app):
        """Whether all read methods are in :const:`ReadMethod.MEM_METHODS`."""
        methods = set(t.metrics.data_read_method for s in app.stages[1:]
                      for t in s.successful_tasks)
        for method in methods:
            if method not in cls.MEM_METHODS:
                return False
        return True

    @staticmethod
    def get_apps(*folders):
        """Get logs as parsed Spark applications.

        Args:
            folders (str): folders relative to the data folder. Example:
                ``get_apps('hibench', 'hbsort')`` for `data/hibench/hbsort`.

        Returns:
            generator: Spark Applications
        """
        root = path.join(path.dirname(__file__), '..')
        folder = path.join(root, 'data', *folders)
        return LogParser().parse_folder(folder)

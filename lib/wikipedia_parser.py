"""Parser for Wikipedia application."""
from itertools import chain

from lib.parser import Parser


class WikipediaParser:
    """Parse wikipedia application logs.

    Input sizes are not correct in the log files, so they are hard coded.
    """

    def __init__(self):
        """Initialize parser."""
        self._parser = Parser()

    def get_apps(self):
        """Return all apps and their sizes.

        Returns:
            list: 2-tuples with app and input size in bytes.
        """
        return chain(self.get_profiling(), self.get_target())

    def get_target(self):
        """Return apps and sizes of target set.

        Returns:
            list: 2-tuples with app and input size in bytes.
        """
        apps = self._parser.get_apps('wikipedia', 'target')
        return ((app, 48542876756) for app in apps)

    def get_profiling(self):
        """Return apps and sizes of profiling set.

        Returns:
            list: 2-tuples with app and input size in bytes.
        """
        # single-VM experiment
        apps = self._parser.get_apps('wikipedia', 'profiling', 'one_vm')
        sizes = _get_input_sizes_1vm()
        profiling = [zip(apps, sizes)]

        # Strong-scaling experiment
        all_apps = self._parser.get_apps('wikipedia', 'profiling',
                                         'strong_scaling1')
        # Use 1 slave from 'one_vm' and skip these ones.
        apps = (a for a in all_apps if len(a.slaves) > 1)
        size = 1073741798
        profiling.append((app, size) for app in apps)

        return chain.from_iterable(profiling)


def _get_input_sizes_1vm():
    """Return a generator for input sizes ordered by filename."""
    for size in (134217465, 268425644, 536816741, 1073741798):
        for _ in range(10):
            yield size

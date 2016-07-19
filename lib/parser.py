"""Parse Spark log files."""

import glob

from sparklogstats import LogParser


class Parser:
    """Manage a set of application executions."""

    def __init__(self):
        self.parser = None

    def parse_folder(self, folder):
        """Parse Spark log files.

        Args:
            folder (str): Path relative to the project root.
                (e.g. data/app/profiling)

        Returns:
            generator: Spark application instances.
        """
        self.parser = LogParser()
        files = sorted(glob.glob('../' + folder + '/app-*'))
        return (self._get_app(log) for log in files)

    def _get_app(self, log):
        self.parser.parse_file(log)
        app = self.parser.app
        # Changing hostnames for number of hosts
        app.slaves = len(app.slaves)
        return app

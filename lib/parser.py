import glob

from sparklogstats import LogParser


class Parser:
    """Manage a set of application executions."""

    def __init__(self):
        self.parser = None

    def parse(self, folder):
        """Parse Spark log files.

        :param str folder: Path relative to the project root
            (e.g. data/app/profiling)
        :rtype: generator
        """
        self.parser = LogParser()
        files = glob.glob('../' + folder + '/app-*')
        return (self._parse_log(log) for log in files)

    def _parse_log(self, log):
        self.parser.parse_file(log)
        app = self.parser.app
        app.slaves = len(app.slaves)
        return app

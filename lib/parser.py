import glob

from sparklogstats import LogParser


class Parser:
    """Manage a set of application executions."""

    def __init__(self):
        self.parser = None

    def parse_folder(self, folder):
        """Parse Spark log files.

        :param str folder: Path relative to the project root
            (e.g. data/app/profiling)
        :returns: Spark application objects
        :rtype: generator
        """
        self.parser = LogParser()
        files = sorted(glob.glob('../' + folder + '/app-*'))
        return (self._get_app(log) for log in files)

    def _get_app(self, log):
        self.parser.parse_file(log)
        app = self.parser.app
        app.slaves = len(app.slaves)
        return app

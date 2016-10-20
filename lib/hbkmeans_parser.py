"""Parse HiBench K-means logs."""
import glob
from multiprocessing import Pool
from os import path
from random import shuffle

from sparklogstats import LogParser


class HBKmeansParser:
    """Parse  HiBench K-means logs."""

    #: Minimum number of samples in target set.
    THRESHOLD = 16384000

    @classmethod
    def get_set(cls, samples):
        """Return whether the app belongs to profiling or target set.

        Args:
            samples (int): Total number of samples.
        """
        return 'profiling' if samples < cls.THRESHOLD else 'target'

    @staticmethod
    def _get_logs():
        root = path.join(path.dirname(__file__), '..')
        filenames = path.join(root, 'data', 'hibench', 'kmeans', 'app-*')
        return glob.glob(filenames)

    @staticmethod
    def map(fn):
        """Apply `fn` to all log files using all cores available.

        Args:
            fn (function): Function to be applied to every log file.

        Returns:
            list: Results of fn applied to every app in random order.
        """
        logs = HBKmeansParser._get_logs()
        shuffle(logs)  # last logs are longer
        with Pool() as p:
            return p.map(fn, logs)

    @classmethod
    def get_apps(cls):
        """Return all apps in random order."""
        parser = LogParser()
        return cls.map(parser.parse_file)

    @classmethod
    def get_app(cls):
        """Return only one app.

        This can be useful for checking the number of stages, for example
        """
        log = HBKmeansParser._get_logs()[0]
        return LogParser().parse_file(log)

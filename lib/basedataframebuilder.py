"""The one and only module for the sake of documentation."""
import pandas as pd
from lib.parser import Parser


class BaseDataFrameBuilder:
    """Help building data frames from Spark log files."""

    def __init__(self, threads, stage):
        self._threads = threads
        self._stage = stage

    @staticmethod
    def from_worker_input(pairs_list):
        """Workers and input DataFrame from a list of value pairs."""
        return pd.DataFrame.from_records(pairs_list,
                                         columns=['workers', 'input'])

    def _build_all(self, folder):
        parser = Parser()
        apps = parser.parse_folder(folder)
        records, columns = self._get_records(apps)
        return pd.DataFrame.from_records(records, columns=columns)

    def _get_records(self, apps):
        records = []
        for app in apps:
            records.append((
                app.stages[0].bytes_read,
                app.slaves,
                self._get_duration(app)
            ))
        columns = ('input', 'workers', 'ms')
        return records, columns

    def _get_duration(self, app):
        if self._stage:
            duration = app.stages[self._stage].durations
        else:
            duration = app.duration
        return duration

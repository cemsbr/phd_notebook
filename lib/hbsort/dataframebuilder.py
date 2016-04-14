"""The one and only module for the sake of documentation."""
import pandas as pd
from lib.parser import Parser
from lib.basedataframebuilder import BaseDataFrameBuilder
from math import ceil


class DataFrameBuilder(BaseDataFrameBuilder):
    """Help building data frames from Spark log files."""

    TARGET_SIZES = (3284983900, 32849603895)

    def __init__(self, threads=None, stage=None):
        super().__init__(threads, stage)
        self._folder = 'data/hibench/sort'
        self._all = None
        self._first = None
        self._nonfirst = None

    def _get_all(self):
        if not isinstance(self._all, pd.DataFrame):
            self._all = self._build_all()
        return self._all

    def _build_all(self):
        parser = Parser()
        apps = parser.parse_folder(self._folder)
        records, columns = self._get_records(apps)
        return pd.DataFrame.from_records(records, columns=columns)

    def _get_all_tasks(self):
        if not isinstance(self._all, pd.DataFrame):
            self._first, self._nonfirst = self._build_all_tasks()
        return self._first, self._nonfirst

    def _build_all_tasks(self):
        parser = Parser()
        apps = parser.parse_folder(self._folder)
        first, nonfirst, columns = self._get_task_records(apps)
        return (pd.DataFrame.from_records(first,
                                          columns=columns),
                pd.DataFrame.from_records(nonfirst,
                                          columns=columns))

    def get_target(self, n):
        """The experiment we want to predict.

        :param int n: Experiment 0 or 1 (more data)
        """
        df = self._get_all()
        size = DataFrameBuilder.TARGET_SIZES[n]
        return df[df.input == size]

    def get_profiling(self):
        df = self._get_all()
        size = min(DataFrameBuilder.TARGET_SIZES)
        return df[df.input < size]

    def get_profiling_tasks(self):
        first, nonfirst = self._get_all_tasks()
        size = min(DataFrameBuilder.TARGET_SIZES)
        return (first[first.input < size], nonfirst[nonfirst.input < size])

    def get_target_tasks(self, n):
        first, nonfirst = self._get_all_tasks()
        size = DataFrameBuilder.TARGET_SIZES[n]
        return (first[first.input == size], nonfirst[nonfirst.input == size])

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

    def _get_task_records(self, apps):
        first, nonfirst = [], []
        for app in apps:
            workers = app.slaves
            size = app.stages[0].bytes_read
            threads = self._threads * workers
            tasks = app.stages[self._stage].tasks
            for task in tasks[:threads]:
                first.append((size, workers, task.duration))
            for task in tasks[threads:]:
                nonfirst.append((size, workers, task.duration))
        return first, nonfirst, ('input', 'workers', 'ms')

    def _get_stage_features(self, app, stage):
        tasks = app.stages[stage].total_tasks
        threads = self._threads
        if tasks > 0:
            iter_f = 1
            if tasks > threads:
                iter_nf = ceil((tasks - threads) / threads)
            else:
                iter_nf = 0
        else:
            iter_f = iter_nf = 0
        return iter_f, iter_nf

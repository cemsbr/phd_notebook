"""The one and only module for the sake of documentation."""
import pandas as pd
from lib import BaseDataFrameBuilder
from lib.parser import Parser


class DataFrameBuilder(BaseDataFrameBuilder):
    """Help building data frames from Spark log files."""

    def __init__(self, threads=None, stage=None):
        super().__init__(threads, stage)
        self._folder = None
        self._repetitions = None
        self._uniq_sizes = None  # unique values

    @staticmethod
    def from_worker_input(pairs_list):
        """Workers and input DataFrame from a list of value pairs."""
        return pd.DataFrame.from_records(pairs_list,
                                         columns=['workers', 'input'])

    def get_target(self, stage=None):
        """The experiment we want to predict."""
        self._init_target(stage)
        return self._build()

    def get_target_tasks(self, stage, threads):
        """The experiment we want to predict."""
        self._init_target(stage, threads)
        return self._build_first_nonfirst_tasks_df()

    def get_strong_scaling(self, n, stage=None):
        """There are two strong scaling experiments: n = 1 and n = 2."""
        self._init_strong_scaling(n, stage)
        return self._build()

    def get_strong_scaling_tasks(self, n, stage, threads):
        """There are two strong scaling experiments: n = 1 and n = 2."""
        self._init_strong_scaling(n, stage, threads)
        return self._build_first_nonfirst_tasks_df()

    def get_weak_scaling(self, stage=None):
        """1 VM with 1 GB, 2 with 2 GB, 4 with 4 GB, ..."""
        self._init_weak_scaling(stage)
        return self._build()

    def get_1vm(self, stage=None):
        """Experiment with one VM and several input sizes."""
        self._init_1vm(stage)
        return self._build()

    def get_1vm_tasks(self, stage, threads):
        """Experiment with one VM and several input sizes."""
        self._init_1vm(stage, threads)
        return self._build_first_nonfirst_tasks_df()

    def _init(self, folder, sizes, reps, stage, threads):
        self._folder = folder
        self._uniq_sizes = sizes
        self._repetitions = reps
        self._stage = stage
        self._threads = threads

    def _init_target(self, stage, threads=None):
        self._init('data/wikipedia/target', [48542876756], 150, stage, threads)

    def _init_1vm(self, stage, threads=None):
        self._init('data/wikipedia/profiling/one_vm', [134217465, 268425644,
                                                       536816741, 1073741798],
                   10, stage, threads)

    def _init_strong_scaling(self, n, stage, threads=None):
        self._init('data/wikipedia/profiling/strong_scaling{:d}'.format(n),
                   [1073741798], 30, stage, threads)

    def _init_weak_scaling(self, stage, threads=None):
        self._init('data/wikipedia/profiling/weak_scaling',
                   [1073741798, 2147481045, 4294959976, 8589900209,
                    17179859955, 34359738259, 48542876756], 30, stage, threads)

    def _get_app_duration(self, app):
        if self._stage is None:
            duration = app.duration
        else:
            duration = app.stages[self._stage].duration
        return duration

    def _get_first_nonfirst_tasks_durations(self, app):
        durations = self._get_task_durations(app)
        threads = app.slaves * self._threads
        first = durations[:threads]
        nonfirst = durations[threads:]
        return (first, nonfirst)

    def _get_task_durations(self, app):
        tasks = app.stages[self._stage].tasks
        for prev, cur in zip(tasks[:-1], tasks[1:]):
            assert prev.start <= cur.start
        return [t.duration for t in tasks if not t.failed]

    def _build_first_nonfirst_tasks(self):
        durations = {'first': [], 'nonfirst': []}
        sizes = {'first': [], 'nonfirst': []}
        workers = {'first': [], 'nonfirst': []}
        parser = Parser()
        apps = parser.parse_folder(self._folder)
        apps_sizes = self._get_app_sizes()
        for app, size in zip(apps, apps_sizes):
            first, nonfirst = self._get_first_nonfirst_tasks_durations(app)
            durations['first'].extend(first)
            durations['nonfirst'].extend(nonfirst)
            sizes['first'].extend([size] * len(first))
            sizes['nonfirst'].extend([size] * len(nonfirst))
            workers['first'].extend([app.slaves] * len(first))
            workers['nonfirst'].extend([app.slaves] * len(nonfirst))
        first = _get_df(workers, sizes, durations, 'first')
        nonfirst = _get_df(workers, sizes, durations, 'nonfirst')
        return first, nonfirst

    def _build(self):
        all_durs, all_sizes, all_workers = [], [], []
        parser = Parser()
        apps = parser.parse_folder(self._folder)
        sizes = self._get_app_sizes()
        for app, size in zip(apps, sizes):
            duration = self._get_app_duration(app)
            all_durs.append(duration)
            all_sizes.append(size)
            all_workers.append(app.slaves)
        return _get_df(all_workers, all_sizes, all_durs)

    def _get_app_sizes(self):
        for size in self._uniq_sizes:
            for _ in range(self._repetitions):
                yield size


def _get_df(workers, sizes, ms, key=None):
    if key:
        workers, sizes, ms = workers[key], sizes[key], ms[key]
    return pd.DataFrame(
        {'workers': workers,
         'input': sizes,
         'ms': ms},
        columns=['workers', 'input', 'ms'])

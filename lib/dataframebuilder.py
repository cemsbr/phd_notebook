"""The one and only module for the sake of documentation."""
import pandas as pd
from lib.parser import Parser


class DataFrameBuilder:
    """Help building data frames from experiment results."""

    @staticmethod
    def get_target_df():
        folder = 'data/wikipedia/target'
        uniq_sizes = [48542876756]
        repetitions = 150
        return _get_df_with_input_size(folder, uniq_sizes, repetitions)

    @staticmethod
    def get_strong_scaling_df(n):
        """There are two strong scaling experiments: n = 1 and n = 2."""
        folder = 'data/wikipedia/profiling/strong_scaling{:d}'.format(n)
        uniq_sizes = [1073741798]
        repetitions = 30
        return _get_df_with_input_size(folder, uniq_sizes, repetitions)

    @staticmethod
    def get_one_vm_df():
        """Experiment with one VM and several input sizes."""
        folder = 'data/wikipedia/profiling/one_vm'
        uniq_sizes = [134217465, 268425644, 536816741, 1073741798]
        repetitions = 10
        return _get_df_with_input_size(folder, uniq_sizes, repetitions)

    @staticmethod
    def get_weak_scaling_df():
        """1 VM with 1 GB, 2 with 2 GB, 4 with 4 GB, ..."""
        folder = 'data/wikipedia/profiling/weak_scaling'
        uniq_sizes = [1073741798, 2147481045, 4294959976, 8589900209,
                      17179859955, 34359738259, 48542876756]
        repetitions = 30
        df = _get_worker_duration_df(folder)
        df['input'] = uniq_sizes * repetitions
        return df


def _get_df_with_input_size(folder, uniq_sizes, repetitions):
    df = _get_worker_duration_df(folder)
    df['input'] = _get_all_sizes(uniq_sizes, repetitions)
    return df


def _get_worker_duration_df(folder):
    parser = Parser()
    apps = parser.parse(folder)
    apps = sorted(apps, key=lambda a: a.start)
    return pd.DataFrame({
        'workers': [app.slaves for app in apps],
        'ms': [app.duration for app in apps]  # duration in milliseconds)
    })


def _get_all_sizes(uniq_sizes, repetitions):
    exp_sizes = []
    for size in uniq_sizes:
        exp_sizes.extend([size] * repetitions)
    return exp_sizes

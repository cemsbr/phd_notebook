from math import ceil
from lib.parser import Parser


class ExperimentInfo:
    SS1 = 'data/wikipedia/profiling/strong_scaling1'
    SS2 = 'data/wikipedia/profiling/strong_scaling2'
    ONE = 'data/wikipedia/profiling/one_vm'

    def __init__(self, threads, block_size, path=None):
        self.threads = threads
        self.block_size = block_size
        self._path = path
        self._parser = Parser()

    def get_n_blocks(self, input_size):
        return ceil(input_size / self.block_size)

    def get_n_tasks(self, stage):
        task_amounts = self._get_ns_tasks(stage)
        assert len(task_amounts) == 1
        return task_amounts[0]

    def _get_ns_tasks(self, stage):
        amounts = set()
        for app in self._get_apps():
            amounts.add(len(app.stages[stage].tasks))
        return list(amounts)

    def _get_apps(self):
        return self._parser.parse_folder(self._path)

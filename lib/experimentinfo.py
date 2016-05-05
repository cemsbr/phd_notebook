from math import ceil
from lib.parser import Parser
import numpy as np


# TODO Path-based
class ExperimentInfo:
    def __init__(self, threads, block_size, path=None):
        self.threads = threads
        self.block_size = block_size
        self._path = path
        self._parser = Parser()

    def get_n_blocks(self, input_size):
        return np.ceil(input_size / self.block_size)

    def get_n_tasks(self, stage):
        task_amounts = self._get_ns_tasks(stage)
        assert len(task_amounts) == 1, 'More than 1 value: {}'.format(
            task_amounts)
        return task_amounts[0]

    def _get_ns_tasks(self, stage):
        amounts = set()
        for app in self._get_apps():
            tasks = app.stages[stage].tasks
            n = sum(1 for t in tasks if not t.failed)
            amounts.add(len(app.stages[stage].tasks))
        return list(amounts)

    def _get_apps(self):
        return self._parser.parse_folder(self._path)

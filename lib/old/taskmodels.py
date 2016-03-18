# pylint: disable=missing-docstring
import numpy as np


class TaskModels:
    def __init__(self, data_size, block_size, threads):
        self.n_tasks = np.ceil(data_size/block_size)
        self.threads = threads

    def get_constant_model(self, duration, n_tasks):
        """Based on task duration mean and number of workers.

        The total execution time is the task duration multiplied by the
        (highest) number of tasks per worker.
        """
        if n_tasks is None:
            n_tasks = self.n_tasks

        def predict(nodes):
            workers = nodes * self.threads
            return np.ceil(n_tasks/workers) * duration

        return predict

    def get_two_mean_model(self, poly_first, poly_nonfirst):
        def predict(nodes):
            first = np.polyval(poly_first, nodes)
            remaining_tasks = self.n_tasks - nodes * self.threads
            nonfirst = self.get_poly_model(poly_nonfirst, remaining_tasks)
            return first + nonfirst(nodes)

        return predict

    def get_poly_model(self, poly, n_tasks=None):
        """
        :param poly: Numpy's polynomial where x is the number of nodes and f(x)
                     is the task duration.
        """
        def predict(nodes):
            duration = np.polyval(poly, nodes)
            const_model = self.get_constant_model(duration, n_tasks)
            return const_model(nodes)

        return predict

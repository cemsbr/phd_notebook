from numpy import ceil


class TaskMeanModel:
    def __init__(self, model, prof_tasks, target_tasks, threads):
        """Adjust number of nodes to preserve first/non-first tasks ratio.

        :param model: A ML model
        :param int prof_tasks: Total number of tasks in profiling phase
        :param int target_tasks: Total number of tasks of prediction
        """
        self._model = model
        self._prof_tasks = prof_tasks
        self._target_tasks = target_tasks
        self._threads = threads

    def adjust_nodes(self, df):
        _df = df.copy()
        _df.workers *= self._target_tasks/self._prof_tasks
        return _df

    def fit(self, df):
        _df = self.adjust_nodes(df)
        self._model.fit(_df)

    def predict(self, df):
        task_means = self._model.predict(df)
        total_threads = df.workers * self._threads
        iterations = ceil(self._target_tasks / total_threads)
        return iterations * task_means

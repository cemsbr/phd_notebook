from numpy import ceil


class TaskMeanModel:
    def __init__(self, model, threads, block_size):
        self._model = model
        self._threads = threads
        self._block_size = block_size

    def predict(self, df):
        task_mean = self._model.predict(df)
        total_threads = df.workers * self._threads
        blocks = ceil(df.input / self._block_size)
        iterations = ceil(blocks / total_threads)
        return iterations * task_mean

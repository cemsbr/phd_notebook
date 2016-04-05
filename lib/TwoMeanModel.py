from math import ceil


class TwoMeanModel:
    def __init__(self,
                 first_model,
                 nonfirst_model,
                 threads,
                 tasks=None,
                 block_size=None):
        self._first_model = first_model
        self._nonfirst_model = nonfirst_model
        self._threads = threads
        self._tasks = tasks
        self._block_size = block_size  # bytes

    def predict(self, df):
        """Prediction based on number of workers and input size.

        :param df: DataFrame with workers and input cols or list of such pairs.
        """
        predictions = []
        for i in range(len(df)):
            workers, size = df.iloc[i]
            tasks = self._get_tasks(size)
            threads = workers * self._threads

            first_mean = self._first_model.predict(df[i:i+1])[0]
            total_time = first_mean

            if tasks > threads:
                nonfirst_tasks = (tasks - threads)
                iterations = ceil(nonfirst_tasks / threads)
                nonfirst_mean = self._nonfirst_model.predict(df[i:i+1])[0]
                total_time += iterations * nonfirst_mean

            predictions.append(total_time)

        return predictions

    def _get_tasks(self, size):
        if self._tasks:
            tasks = self._tasks
        else:
            tasks = ceil(size / self._block_size)
        return tasks

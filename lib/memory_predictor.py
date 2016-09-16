"""Predict if data fits in the memory."""
import pandas as pd


class MemoryPredictor:
    """Predict if data fits in the memory."""

    def __init__(self, train_df):
        """Require training csv filename.

        Args:
            train_df (pd.DataFrame): Profiling phase of all apps.
        """
        self._thresholds = _predict(train_df)

    def fits_in_memory(self, app, input_size, workers):
        """Predict whether data fits in memory or not.

        Args:
            app (str): Application name.
            input_size (int): Input size (same unit as training data).
            workers (int): Number of workers (same unit as training data).

        Returns:
            boolean: Whether the input can be processed entirely from the
                memory.
        """
        if app not in self._thresholds:
            return True
        else:
            return input_size / workers < self._thresholds[app]

    def filter_in_memory(self, df):
        """Remove cases when data does not fit in the memory.

        The column `in_memory` is removed and prediction is used.

        Args:
            df (pd.DataFrame): DataFrame to filter.
        """
        in_memory = df.apply(predict(self), axis=1)
        return df[in_memory].drop('in_memory', axis=1)


def _predict(train_df):
    """Minimum data size per worker that does not fit in the memory.

    Returns:
        dict: key is application name, value is maximum input per worker that
            fits in the memory.
    """
    no_mem = train_df[~train_df.in_memory]
    worker_input = pd.DataFrame({'app': no_mem.application,
                                 'input': no_mem.input / no_mem.workers})
    threshold = worker_input.groupby('app', as_index=False).min()
    return {row[0]: row[1] for row in threshold.get_values()}


class Report:
    """Check :class:`.MemoryPredictor accuracy."""

    def __init__(self, predictor):
        """Evaluate a predictor.

        Args:
            predictor (MemoryPredictor): Trained predictor to be evaluated.
        """
        self._pred = predictor

    def get_stats(self, df):
        """Report correct and wrong predictions.

        Args:
            csv_file (str): CSV filename that includes correct answers.

        Returns:
            tuple: Number of cases when data fits in the memory, does not fit
                and number of errors.
        """
        answer = df.in_memory
        prediction = df.apply(predict(self._pred), axis=1)
        is_correct = prediction == answer
        return (answer[answer].count(),           # fit in the memory
                answer[~answer].count(),          # does not fit
                is_correct[~is_correct].count())  # errors

    def get_text(self, title, df):
        """Return a string ready to be printed."""
        return 'Results for {}: {} cases fit in the memory, {} do not,' \
               ' {} prediction errors.'.format(title, *self.get_stats(df))


def predict(predictor):
    """Return a function that receives only a DataFrame."""
    def predict_df(df):
        """Return predictor answer."""
        return predictor.fits_in_memory(df.application, df.input, df.workers)
    return predict_df

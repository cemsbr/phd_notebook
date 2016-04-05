"""The one and only module for the sake of documentation."""
from pandas import DataFrame


class OneManyModel:
    """Uses an experiment from 1 VM and another with different amounts."""

    def __init__(self, one_vm_model, many_vms_model):
        self._one = one_vm_model
        self._many = many_vms_model
        self._many_intercept_bak = many_vms_model.get_linreg().intercept_
        self._size_per_worker = None

    def predict(self, df):
        """Prediction based on number of workers and input size.

        :param df: DataFrame with workers and input cols or list of such pairs.
        """
        if isinstance(df, list):
            pairs = df
        else:
            pairs = df[['workers', 'input']].values
        return [self._predict(row) for row in pairs]

    def _predict(self, features):
        workers, size = features
        size_per_worker = size  # / workers
        self._set_intercept(size_per_worker)
        return self._many.predict([[workers]])[0]

    def _set_intercept(self, size_per_worker):
        if self._size_per_worker != size_per_worker:
            self._update_intercept(size_per_worker)
            self._size_per_worker = size_per_worker

    def _update_intercept(self, size_per_worker):
        df = DataFrame({'workers': [1], 'input': [size_per_worker]})
        df['ms'] = self._one.predict(df.drop('workers', axis=1))
        self._many.set_intercept(df.drop('input', axis=1))

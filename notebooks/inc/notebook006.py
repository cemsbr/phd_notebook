"""Include file for Jupyter notebook number 006."""
import numpy as np
import pandas as pd
import re
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold

from lib.bundler import Bundler


Bundler.set_bundles_root('..', '..', 'bundles')

__all__ = ('Bundler', 'get_prediction_cols', 'pd', 'Predictor')

def get_prediction_cols(df, stage_fn):
    """Apply *stage_fn* to *df*. *stage_fn* receives df and stage column."""
    p = re.compile('^s\d+$')
    def is_stage(col):
        return p.match(col)

    # input and workers are used only to split data into training and
    # validation sets
    features_cols = ['application', 'set', 'input', 'workers', 'duration_ms']
    features = df[features_cols].copy()

    for col in df:
        if is_stage(col):
            features[col] = stage_fn(df, col)

    return features


class Predictor:
    """Predict target set from profiling set.

    Make sure to set *df* attribute (:method:`set_df`).
    """

    def __init__(self, df=None):
        """Constructor.

        Args:
            pd.DataFrame: Contains features, application, set and duration_ms
                columns. Everything is calculated from this attribute.

        See also:
            :meth:`set_df`
        """
        if df is not None:
            self.set_df(df)

    def set_df(self, df):
        """Set which DataFrame to use.

        Args:
            pd.DataFrame: Contains features, application, set and duration_ms
                columns. Everything is calculated from this attribute.
        """
        self._df = df
        self._groups = df.apply(lambda df: (df.input, df.workers), axis=1)

    def evaluate(self):
        """Print RMSE in seconds for all apps, profiling and target sets."""
        print('Profiling:')
        for app in self._df.application.unique():
            self._evaluate_profiling(app)
        print('Target:')
        for app in self._df.application.unique():
            self._evaluate_target(app)

    def _evaluate_profiling(self, app, alphas=None):
        profiling = self._get_app_set(app, 'profiling')
        x, y = _get_xy(profiling)

        preds = pd.DataFrame()
        for x_train, y_train, x_test, test_ix in self._split(x, y):
            # Further divide training set into trainining and validation sets
            # to choose regularization value
            lm = self._choose_alpha(x_train, y_train, alphas)
            lm.fit(x_train, y_train)
            pred = pd.DataFrame(lm.predict(x_test), index=test_ix)
            preds = preds.append(pred, verify_integrity=True)  # noqa

        mse = metrics.mean_squared_error(y, preds.sort_index())
        print('- {} profiling RMSE = {:.2f} sec'.format(app, mse**0.5 / 1000))

    def _split(self, x, y):
        cv = GroupKFold(n_splits=3)
        groups = self._groups.loc[x.index]
        for train_ix, test_ix in cv.split(x, groups=groups):
            # train_ix and test_ix starts from 0, so we use iloc
            x_train = x.iloc[train_ix]
            y_train = y.iloc[train_ix]
            x_test = x.iloc[test_ix]
            yield x_train, y_train, x_test, test_ix

    def _evaluate_target(self, app, alphas=None):
        # Training with profiling data
        df = self._get_app_set(app, 'profiling')
        x, y = _get_xy(df)
        lm = self._choose_alpha(x, y)
        lm.fit(x, y)

        # Predicting target using lm from above
        df = self._get_app_set(app, 'target')
        x, y = _get_xy(df)
        mse = metrics.mean_squared_error(y, lm.predict(x))

        print('- {} target RMSE = {:.2f} sec'.format(app, mse**0.5 / 1000))

    def _get_app_set(self, app, sset):
        query = (self._df.application == app) & (self._df.set == sset)
        return self._df[query].dropna(axis=1)

    def _choose_alpha(self, x, y, alphas=None):
        """Choose best alpha value for regularization.

        Use GroupKFold where a group is a combination of input size and number
        of workers. The prediction of a group is done when it is out of the
        training set.

        Args:
            alphas (iterable): Defaults to (0.0, 0.1, 0.3, 1.0, 3.0, 10.0).

        Returns:
            The Ridge linear model with the alpha that minimizes RMSE in the
                validation set.
        """
        if alphas is None:
            alphas = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
        cv_mse = {}
        for alpha in alphas:
            lm = Ridge(alpha, normalize=True)
            mse = self._get_estimator_mse(x, y, lm)
            cv_mse[mse] = lm
        min_mse = min(cv_mse.keys())
        best_lm = cv_mse[min_mse]
        return best_lm

    def _get_estimator_mse(self, x, y, estimator):
        """Return the RMSE for *estimator*.

        Use GroupKFold where a group is a combination of input size and number
        of workers. The prediction of a group is done when it is out of the
        training set.
        """
        groups = self._groups.loc[x.index]
        cv = GroupKFold(n_splits=3)
        prediction = cross_val_predict(estimator, x, y, groups, cv)
        return metrics.mean_squared_error(y, prediction)


def _get_xy(df):
    """Return features for *app* of set *sset*."""
    cols2drop = ['application', 'set', 'input', 'workers', 'duration_ms']
    return df.drop(cols2drop, axis=1), df[['duration_ms']]

"""Similiar to scikit's RidgeCV, but use GroupKFold."""
import logging

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Ridge

from lib.utils import safe_log

log = logging.getLogger()


class GroupRidgeCV:
    """Similiar to RidgeCV, but use can use groups."""

    def __init__(self, groups, cv, use_logs=False, alphas=None,
                 fit_intercept=True):
        """Set alpha values.

        Args:
            alphas (iterable): Alpha values to test. Defaults to
                (0, 0.1, 0.3, 1, 3, 10).
        """
        self._groups = groups
        self._cv = cv
        if alphas is None:
            alphas = (0.000001, 0.1, 1, 10)
        self._alphas = alphas
        self._fit_intercept = fit_intercept
        #: Ridge: scikit's Ridge with the alpha chosen by CV.
        self._ridge = None
        self._x, self._y = None, None
        self._use_logs = use_logs

    def get_params(self, deep):
        """Method used by scikit-learn."""
        return {'groups': self._groups, 'cv': self._cv, 'alphas': self._alphas,
                'fit_intercept': self._fit_intercept}

    def fit(self, x, y):
        """Fit Ridge linear model with best alpha from cross validation.

        Data is split using groups set through :meth:`set_groups`.

        Returns:
            float: Cross-validation score.
            float: Cross-validation score in log scale if *use_logs* was used
                in the constructor. Otherwise, this value equals the previous
                one.
        """
        self._x = safe_log(x) if self._use_logs else x
        self._y = safe_log(y) if self._use_logs else y
        cv_scr, cv_scr_log, alpha = self._choose_alpha()
        self._ridge = Ridge(alpha, fit_intercept=self._fit_intercept,
                            normalize=True)
        # With the best alpha chosen by CV, train with all the data.
        self._ridge.fit(self._x, self._y)
        log.debug('Best alpha = %.2f', alpha)
        return cv_scr, cv_scr_log

    def _choose_alpha(self):
        evals = (self._cross_val(alpha) for alpha in self._alphas)
        return sorted(evals)[0]

    def _cross_val(self, alpha):
        lm = Ridge(alpha, fit_intercept=self._fit_intercept, normalize=True)
        y_pred = pd.Series()
        groups = self._groups.loc[self._x.index]
        for train_ix, test_ix in self._cv.split(self._x, self._y, groups):
            s = self._get_test_prediction(lm, train_ix, test_ix)
            y_pred = y_pred.append(s, verify_integrity=True)  # noqa
        y_pred.sort_index(inplace=True)
        scr = metrics.mean_squared_error(self._y, y_pred)
        scr_nolog = self._exp_score(y_pred) if self._use_logs else scr
        return scr_nolog, scr, alpha

    def _get_test_prediction(self, lm, train_ix, test_ix):
        x_train = self._x.iloc[train_ix]
        y_train = self._y.iloc[train_ix]
        x_test = self._x.iloc[test_ix]
        lm.fit(x_train, y_train)
        test_pred = lm.predict(x_test)
        return pd.Series(test_pred, index=x_test.index)

    def _exp_score(self, y_pred):
        y_pred = np.exp(y_pred)
        y_true = np.exp(self._y)
        return metrics.mean_squared_error(y_true, y_pred)

    def predict(self, x):
        """Predict using alpha chosen by :meth:`fit`."""
        y_pred = self._ridge.predict(safe_log(x) if self._use_logs else x)
        return np.exp(y_pred) if self._use_logs else y_pred

    def score(self, x, y):
        """Return mean squared error. Must use :meth:`fit` before."""
        return metrics.mean_squared_error(y, self.predict(x))

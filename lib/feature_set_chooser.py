"""A Model that choose the best feature set using cross validation."""
import logging

import pandas as pd
from sklearn import metrics

from lib.group_ridge_cv import GroupRidgeCV

log = logging.getLogger()


class FeatureSetChooser:
    """Choose the best feature set using cross validation and GroupRidgeCV."""

    def __init__(self, groups, cv):
        """Initialize object attributes."""
        self._cv = cv
        self._groups = groups
        self._x = None
        self._y = None

    def choose_feature_set(self, feature_sets):
        """Return the best feature set's name using cross validation.

        Select the feature set with lowest MSE. Data is split using groups set
        through :meth:`set_groups`.

        Args:
            feature_sets (iterable): Each tuple has feature set name, x and y.

        Returns:
            str: Chosen column set name.
            float: Cross-validation score.
            GroupRidgeCV: Trained model.
        """
        return sorted(self._cross_val(fs) for fs in feature_sets)[0]

    def _cross_val(self, feat_set):
        col_set, x, y, use_logs = feat_set
        self._x, self._y = x, y
        lm = GroupRidgeCV(self._groups, self._cv, use_logs)
        groups = self._groups.loc[x.index]
        y_pred = pd.Series()
        for train_ix, test_ix in self._cv.split(x, y, groups):
            s = self._get_test_prediction(lm, train_ix, test_ix)
            y_pred = y_pred.append(s, verify_integrity=True)  # noqa
        y_pred.sort_index(inplace=True)
        score = metrics.mean_squared_error(y, y_pred)
        lm.fit(x, y)
        return score, col_set, lm

    def _get_test_prediction(self, lm, train_ix, test_ix):
        x_train = self._x.iloc[train_ix]
        y_train = self._y.iloc[train_ix]
        x_test = self._x.iloc[test_ix]
        lm.fit(x_train, y_train)
        test_pred = lm.predict(x_test)
        return pd.Series(test_pred, index=x_test.index)

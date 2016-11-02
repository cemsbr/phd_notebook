"""Base class to choose best alternative using group cross validation."""
from abc import abstractmethod

import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut


class GroupCrossValidation:
    """Choose best alteranative using CV with GroupKFold."""

    _groups = None

    def __init__(self):
        """Initialize GroupKFold with 3 splits."""
        # self._cv = GroupKFold(n_splits=3)
        self._cv = LeaveOneGroupOut()

    @classmethod
    def set_groups(cls, groups):
        """Set all groups to be further selected by index.

        Args:
            groups (pd.Series): Groups for splitting data.
        """
        cls._groups = groups

    def _choose_best(self):
        """Return the col set with best score from :meth:`_cross_validate`.

        Returns:
            float: The best score found.
            str: Column set name.
            Trained linear model.
        """
        return sorted(self._cross_validate())[0]

    @abstractmethod
    def _cross_validate(self):
        """Return tuples of error to minimize and associated item.

        Returns:
            iterable: (error, item) generator, list or tuple.
        """
        pass

    def _cross_val_error(self, lm, x, y, groups, is_log=False):
        y_pred = cross_val_predict(lm, x, y, groups, self._cv)
        if is_log:
            y = np.exp(y)
            y_pred = np.exp(y_pred)
        return metrics.mean_squared_error(y, y_pred)

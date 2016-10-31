"""Base class to choose best alternative using group cross validation."""
from abc import abstractmethod

from sklearn import metrics
from sklearn.model_selection import cross_val_predict, GroupKFold


class GroupCrossValidation:
    """Choose best alteranative using CV with GroupKFold."""

    _groups = None

    def __init__(self):
        """Initialize GroupKFold with 3 splits."""
        self._cv = GroupKFold(n_splits=3)

    @classmethod
    def set_groups(cls, groups):
        """Set all groups to be further selected by index.

        Args:
            groups (pd.Series): Groups for splitting data.
        """
        cls._groups = groups

    def _choose_best(self):
        """Return the item with minimum error from :meth:`_cross_validate`."""
        mse_item = sorted(self._cross_validate())
        return mse_item[0][1]  # Item of the first mse_item (minimum mse)

    @abstractmethod
    def _cross_validate(self):
        """Return tuples of error to minimize and associated item.

        Returns:
            iterable: (error, item) generator, list or tuple.
        """
        pass

    def _cross_val_error(self, lm, x, y, groups):
        y_pred = cross_val_predict(lm, x, y, groups, self._cv)
        return metrics.mean_squared_error(y, y_pred)

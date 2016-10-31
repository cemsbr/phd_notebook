"""A Model that choose the best feature set using cross validation."""
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from lib.group_ridge_cv import GroupRidgeCV
from lib.group_cross_val import GroupCrossValidation


class FeatureSetChooser(GroupCrossValidation):
    """Choose the best feature set using cross validation and GroupRidgeCV."""

    def __init__(self):
        """Initialize object attributes."""
        super().__init__()
        self._feature_sets = None

    @classmethod
    def set_groups(cls, groups):
        """Set all groups to be further selected by index.

        It will also set GroupRidgeCV groups.

        Args:
            groups (pd.Series): Groups for splitting data.
        """
        super().set_groups(groups)
        GroupRidgeCV.set_groups(groups)

    def choose_feature_set(self, feature_sets):
        """Return the best feature set's name using cross validation.

        Select the feature set with lowest MSE. Data is split using groups set
        through :meth:`set_groups`.

        Args:
            feature_sets (iterable): Each tuple has feature set name, x and y.

        Returns:
            str: Chosen column set name.
            GroupRidgeCV: Trained model.
        """
        self._feature_sets = feature_sets
        return self._choose_best()

    def _cross_validate(self):
        """Return error and a tuple with column set name and trained model."""
        groups = None
        for col_set, x, y, is_log in self._feature_sets:
            if groups is None:
                groups = self._groups.loc[x.index]
            lm = GroupRidgeCV()
            # Train with 2/3 and predict 1/3 of the groups.
            mse = self._cross_val_error(lm, x, y, groups, is_log)
            rmse = mse**0.5 / 1000
            print(col_set, rmse)
            lm.fit(x, y)
            yield (mse, (col_set, lm))

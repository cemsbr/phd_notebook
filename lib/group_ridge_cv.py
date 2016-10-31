"""Similiar to scikit's RidgeCV, but use GroupKFold."""
from sklearn import metrics
from sklearn.linear_model import Ridge

from lib.group_cross_val import GroupCrossValidation


class GroupRidgeCV(GroupCrossValidation):
    """Similiar to RidgeCV, but use GroupKFold."""

    def __init__(self, alphas=None):
        """Set alpha values.

        Args:

            alphas (iterable): Alpha values to test. Defaults to
                (0, 0.1, 0.3, 1, 3, 10).
        """
        super().__init__()
        if alphas is None:
            alphas = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
        self._alphas = alphas
        #: Ridge: scikit's Ridge with the alpha chosen by CV.
        self._ridge = None
        self._x, self._y = None, None

    def get_params(self, deep):
        """Method used by scikit-learn."""
        return {'alphas': self._alphas}

    def fit(self, x, y):
        """Fit Ridge linear model with best alpha from cross validation.

        Data is split using groups set through :meth:`set_groups`.
        """
        self._x, self._y = x, y
        alpha = self._choose_best()
        self._ridge = Ridge(alpha, normalize=True)
        # With the best alpha chosen by CV, train with all the data.
        self._ridge.fit(x, y)

    def _cross_validate(self):
        groups = self._groups.loc[self._x.index]
        for alpha in self._alphas:
            ridge = Ridge(alpha, normalize=True)
            # Train with 2/3 and predict 1/3 of the groups.
            mse = self._cross_val_error(ridge, self._x, self._y, groups)
            yield (mse, alpha)

    def predict(self, x):
        """Predict using alpha chosen by :meth:`fit`."""
        return self._ridge.predict(x)

    def score(self, x, y):
        """Return mean squared error. Must use :meth:`fit` before."""
        return metrics.mean_squared_error(y, self.predict(x))

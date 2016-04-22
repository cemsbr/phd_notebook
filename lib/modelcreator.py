from itertools import chain, combinations
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class ModelCreator:
    def __init__(self, linear_models, features, degrees=(1, )):
        self._linear_models = linear_models
        self._features = features
        self._degrees = degrees

    def get_models(self):
        for feat_set in _powerset(self._features):
            for degree in self._degrees:
                for model in self._linear_models:
                    yield Model.copy_from(model, degree, feat_set)


class Model:
    def __init__(self, linear_model, degree, features):
        self._model = _get_poly(linear_model, degree)
        self._features = features

    @classmethod
    def copy_from(cls, linear_model, degree, features):
        linear_copy = _copy_model(linear_model)
        return cls(linear_copy, degree, features)

    def fit(self, x, y):
        feat_df = self._calc_features(x)
        self._model.fit(feat_df, y)

    def score(self, x, y):
        feat_df = self._calc_features(x)
        predictions = self._model.predict(feat_df)
        errors = {}
        for name, fn in (('RMSE', _rmse), ('MPE', _mpe), ('MAE', _mae)):
            errors[name] = fn(predictions, y)
        return errors

    def _calc_features(self, df):
        new = pd.DataFrame()
        for feature in self._features:
            col, fn = feature
            new[col] = fn(df)
        return new


def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(1, len(s) + 1))


def _copy_model(linear):
    copy = linear.__class__()
    copy.set_params(**linear.get_params())
    return copy


def _get_poly(linear, degree):
    if degree > 1:
        poly = PolynomialFeatures(degree=degree)
        model = Pipeline([('poly', poly), ('linear', linear)])
    else:
        model = linear
    return model


def _rmse(pred, y):
    """Root Mean Squared Error."""
    return np.sqrt(sum((y - pred)**2) / len(y))


def _mpe(pred, y):
    """Mean Percentage Error."""
    return (np.abs(y - pred) / y).mean()


def _mae(pred, y):
    """Mean Absolute Error"""
    return (np.abs(y - pred) / len(y)).mean()

from itertools import chain, combinations
import numpy as np
from sklearn import metrics


class ModelCreator:
    def __init__(self, linear_models, features, ycol):
        self._linear_models = linear_models
        self._features = features
        self._ycol = ycol

    def get_models(self):
        for model in self._linear_models:
            for feat_set_tuple in _powerset(self._features):
                feat_set = list(feat_set_tuple)
                yield Model.copy_from(model, feat_set, self._ycol)

    def __len__(self):
        i = 0
        for _ in _powerset(self._features):
            i += len(self._linear_models)
        return i


class Model:
    def __init__(self, linear_model, features, ycol):
        self._model = linear_model
        self._features = features
        self._ycol = ycol
        self.is_log = False

    @classmethod
    def copy_from(cls, linear_model, features, ycol):
        linear_copy = _copy_model(linear_model)
        return cls(linear_copy, features, ycol)

    def fit(self, df):
        x, y = self._split_xy(df)
        self._model.fit(x, y)

    def predict(self, x):
        feat_x = x[self._features]
        pred = self._model.predict(feat_x)
        if self.is_log:
            pred = 2**pred
        return pred

    def score(self, df):
        x, y = self._split_xy(df)
        pred = self.predict(x)
        return {
            # Root Mean Squared Error
            'RMSE': metrics.mean_squared_error(y, pred)**0.5,
            # Mean Absolute Error
            'MAE': metrics.mean_absolute_error(y, pred),
            # Mean Percentage Error
            'MPE': (np.abs(y - pred) / y).mean(),
        }

    def _split_xy(self, df):
        return df[self._features], df[self._ycol]

    def humanize(self):
        human = {'log': self.is_log}
        params = {}

        model = self._model
        all_params = model.get_params()
        cls = model.__class__.__name__
        if cls == 'LinearRegression':
            human['model'] = 'LinReg'
        elif cls == 'RidgeCV':
            human['model'] = cls
            params['alphas'] = all_params['alphas']
            try:
                params['best'] = model.alpha_
            except AttributeError:
                params['best'] = None
        else:
            human['model'] = cls
            params.update(all_params)

        human['features'] = self._features
        human['nr_feats'] = len(self._features)
        human['params'] = ', '.join('{}: {}'.format(k, params[k])
                                    for k in sorted(params))
        return human

    def __str__(self):
        h = self.humanize()
        return '      linear model: {}\n' \
            '      duration log: {}\n' \
            '            params: {}\n' \
            'number of features: {}\n' \
            '          features: {}'.format(h['model'], h['log'], h['params'],
            h['nr_feats'], ', '.join(h['features']))


def _powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r + 1) for r in range(len(s)))


def _copy_model(linear):
    copy = linear.__class__()
    copy.set_params(**linear.get_params())
    return copy

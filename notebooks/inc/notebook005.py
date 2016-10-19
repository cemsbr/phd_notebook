"""Include file for Jupyter notebook number 005."""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import (cross_val_predict, GroupKFold)

import inc.plt_config  # noqa
from lib.bundler import Bundler
from lib.plotter import Plotter

__all__ = ('Bundler', 'evaluate_feature_sets', 'np', 'pd', 'plot_all',
           'Predictor')

Bundler.set_bundles_root('..', '..', 'bundles')


def plot_wikipedia(predictor, save_pdf=False):
    """Plot wikipedia target prediction and real values."""
    target, model = predictor.get_plot_data('wikipedia')
    output = '/tmp/wikipedia.pdf' if save_pdf else None
    plotter = Plotter(xlim=(3, 65), ylim=(0, 200))
    plotter.plot_model(model, target, output)
    print('Prediction of the Wikipedia application target execution duration.')


def _get_outputs(save_pdf, *basenames):
    return ['/tmp/' + f for f in basenames] if save_pdf else (None, None)


def plot_hbkmeans(predictor, save_pdf=False):
    """Plot actual results and model's predictions."""
    target, model = predictor.get_plot_data('hbkmeans')
    outputs = _get_outputs(save_pdf, 'hbkmeans16.pdf', 'hbkmeans65.pdf')
    plotter = Plotter(xlim=(7.5, 32.5))
    plotter.plot_model(model, target[target.input == 16384000], outputs[0])
    plotter = Plotter(xlim=(30, 130))
    plotter.plot_model(model, target[target.input == 65536000], outputs[1])
    print('The top figure is the result of the HiBench K-means application'
          ' with 16,384,000 samples. The second figure uses 65,536,000'
          ' samples.')


def plot_hbsort(predictor, save_pdf=False):
    """Plot actual results and model's predictions."""
    target, model = predictor.get_plot_data('hbsort')
    outputs = _get_outputs(save_pdf, 'hbsort3.pdf', 'hbsort30.pdf')
    plotter = Plotter(xlim=(0.75, 16.25))
    plotter.plot_model(model, target[target.input < 15 * 1024**3], outputs[0])
    plotter = Plotter(xlim=(14, 130))
    plotter.plot_model(model, target[target.input > 15 * 1024**3], outputs[1])
    print('The top figure is the result of the HiBench Sort application with'
          ' 3-GB input. The second figure uses 31-GB of data.')


def plot_all(predictor):
    """Plot predictions and real results for all target experiments."""
    plot_wikipedia(predictor)
    plot_hbsort(predictor)
    plot_hbkmeans(predictor)


def evaluate_feature_sets(feature_sets):
    predictor = Predictor()
    app_results = {}
    for feature_set in feature_sets:
        use_log, features = feature_set[0], feature_set[1:]
        predictor.use_log = use_log
        predictor.set_features(features)
        # Reindexing so app_results levels are: app, set, feature set RMSE.
        # ar mans App RMSE
        # p and t mean Profiling and Target
        for ar_p, ar_t in zip(*predictor.get_rmse()):
            assert ar_p[0] == ar_t[0]
            app = ar_p[0]
            feat_set_name = ', '.join(t[0] for t in features)
            if use_log:
                feat_set_name += ', log(y)'
            app_lvl = app_results.setdefault(app, {})
            for sset, rmse in (('profiling', ar_p[1]), ('target', ar_t[1])):
                set_lvl = app_lvl.setdefault(sset, [])
                set_lvl.append((rmse, feat_set_name))
    # Sorting by RMSE in each app in each set
    for app in sorted(app_results):
        print('{}:'.format(app))
        for sset in sorted(app_results[app]):
            print('- {}:'.format(sset))
            rmse_feat = app_results[app][sset]
            rmse_feat.sort()
            for rmse, features in rmse_feat:
                print('  - {:.2f} sec: {}'.format(rmse / 1000, features))



class Predictor:
    """Predict application duration based on the profiling set.

    Use data without outliers from *outlier_detection* bundle.
    """

    def __init__(self, use_log=False):
        """Load data from experiments without outliers."""
        csv_file = Bundler.get_bundle('outlier_detection').get_file(
            'no_outliers.csv')
        self.use_log = use_log
        #: Experiment DataFrame
        self._xp_df = pd.read_csv(csv_file)
        #: Groups to separate combinations of input and workers
        self._groups = self._xp_df.apply(lambda df: (df.input, df.workers),
                                         axis=1)
        #: DataFrame with features and *y* values to be used in prediction
        self._df = None
        self._features = None

    def set_features(self, features):
        """Set features for prediction.

        Args:
            iterable: Features to use in the prediction. Contains tuples with
                feature names and a function that receives experiment DataFrame
                and returns the feature value.
        """
        self._df = pd.DataFrame()
        for name, fn in features:
            self._df[name] = fn(self._xp_df)
        self._features = features

    def get_features(self):
        """Return features set by user."""
        return self._features

    def get_plot_data(self, app):
        """Return data for plotting *app* real and prediction results.

        Returns:
            pd.DataFrame: input (bytes), workers and duration (ms) from
                target set.
            sklearn.linear_model.Ridge: Linear model for prediction.
        """
        df = self._xp_df
        query = (df.application == app) & (df.set == 'target')
        cols = ['input', 'workers', 'duration_ms']
        target = df[query][cols]

        x, y = self._get_app_data(app, 'profiling')
        lm = self._choose_alpha(x, y)
        lm.fit(x, y)

        return target, Model(lm, self._features, self.use_log)

    def print_rmse(self):
        """Print RMSE in seconds for each application of profiling set.

        Must use :method:`set_features` before.
        """
        def print_set_rmse(set_name, app_rmse):
            """Print RMSE for a set (profiling or target)."""
            print(set_name, 'results:')
            for app, rmse in app_rmse:
                print("- {} RMSE = {:.2f} sec".format(app, rmse / 1000))
            rmses = [t[1] for t in app_rmse]
            print("- Mean: {:.2f} sec".format(np.mean(rmses) / 1000))
            print("- Max: {:.2f} sec".format(max(rmses) / 1000))

        profiling, target = self.get_rmse()
        print_set_rmse('Profiling', profiling)
        print_set_rmse('Target', target)

    def get_rmse(self):
        """Return profiling and target RMSE values for each app.

        Returns:
            list: Tuples of the form (app, rmse).
        """
        profiling, target = [], []
        for app in self._xp_df.application.unique():
            x, y = self._get_app_data(app, 'profiling')

            mse = self._get_mse_profiling(x, y)
            profiling.append((app, mse**0.5))

            mse = self._get_mse_target(x, y, app)
            target.append((app, mse**0.5))

        return profiling, target

    def _get_app_data(self, app, sset):
        """Return x, y and groups for *app* and *sset* set."""
        df = self._xp_df
        indexes = (df.application == app) & (df.set == sset)
        x = self._df[indexes]
        y = self._xp_df[indexes][['duration_ms']]
        if self.use_log:
            y = np.log2(y)
        return x, y

    def _get_mse_profiling(self, x, y, alphas=None):
        """Calculate prediction RMSE.

        Use GroupKFold where a group is a combination of input size and number
        of workers. The prediction of a group is done when it is out of the
        training set.
        """
        # Training set is 2/3 of the data
        groups = self._groups.loc[x.index]
        cv = GroupKFold(n_splits=3)
        preds = None

        for train_ix, test_ix in cv.split(x, groups=groups):
            # train_ix and test_ix starts from 0, so we use iloc
            x_train, y_train = x.iloc[train_ix], y.iloc[train_ix]
            x_test = x.iloc[test_ix]

            # Choose best alpha value for regularization based on training set
            lm = self._choose_alpha(x_train, y_train, alphas)
            lm.fit(x_train, y_train)
            pred = pd.DataFrame(lm.predict(x_test), index=test_ix)
            preds = pred if preds is None else preds.append(
                pred, verify_integrity=True)

        return self._calc_mse(y, preds.sort_index())

    def _get_mse_target(self, x, y, app, alphas=None):
        """Train using profiling and measure RMSE in target.

        Args:
            x (DataFrame): Profiling features for training.
            y (DataFrame): Profiling true values for training.
        """
        lm = self._choose_alpha(x, y)
        lm.fit(x, y)
        target_x, target_y = self._get_app_data(app, 'target')
        return self._calc_mse(target_y, lm.predict(target_x))

    def _choose_alpha(self, x, y, alphas=None):
        """Choose best alpha value for regularization.

        Use GroupKFold where a group is a combination of input size and number
        of workers. The prediction of a group is done when it is out of the
        training set.

        Args:
            alphas (iterable): Defaults to (0.0, 0.1, 0.3, 1.0, 3.0, 10.0).

        Returns:
            The Ridge linear model with the alpha that minimizes RMSE.
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

    def _calc_mse(self, y_true, y_pred):
        if self.use_log:
            y_true, y_pred = 2**y_true, 2**y_pred
        return metrics.mean_squared_error(y_true, y_pred)


class Model:
    """Predict based on input in bytes and workers."""

    def __init__(self, linear_model, features, use_log):
        """Initialize with a trained scikit linear model."""
        self._lm = linear_model
        self._features = features
        self._use_log = use_log

    def predict(self, x):
        """Predict duration with input in bytes and workers."""
        feat_df = self._get_feature_df(x)
        predictions = self._lm.predict(feat_df)
        return 2**predictions if self._use_log else predictions

    def _get_feature_df(self, x):
        df = pd.DataFrame()
        for name, fn in self._features:
            df[name] = fn(x)
        return df

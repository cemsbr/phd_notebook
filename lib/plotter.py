"""The one and only module for the sake of documentation."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.config import Config
from lib.featurecreator import FeatureCreator


class Plotter:
    """Plotting helper."""

    def __init__(self, xlim=None, ylim=(0, None), figsize=(4.5, 3), loc=None,
                 logx=False, logy=False):
        """Set plot parameters."""
        self._cfg = {'xlim': xlim, 'ylim': ylim, 'figsize': figsize,
                     'loc': loc, 'logx': logx, 'logy': logy, 's': None}
        self.ax = None
        self._xcol = None
        self._xlabel = None
        self._labels = None

    def update_config(self, **kwargs):
        """Update plot configuration."""
        for k, v in kwargs.items():
            self._cfg[k] = v

    def plot_outliers(self, df):
        """Plot total durations by number of workers."""
        dfp = self._check_type(df)
        self._plot_outliers(dfp[df.outlier], dfp[~df.outlier])
        self._finalize(dfp)
        plt.show()

    def plot_model(self, model, target, output=None):
        """Compare model and target experiment.

        :param model: lib.Model
        :param target: DataFrame of target experiment (workers, input, ms),
            without outliers
        :param str output: Output filename to save the plot.
        """
        # Generate an X matrix with more workers to smooth model line.
        pred_df = _add_x(target)

        # Adding model prediction column
        pred_df['duration_ms'] = model.predict(pred_df)

        # Plot prediction values
        plt_kwargs = self._get_cfg_kwargs(['figsize', 'logx', 'logy'])
        dfp = self._check_type(pred_df)
        dfp.sort_values(self._xcol, inplace=True)
        self.ax = dfp.plot(self._xcol,
                           'seconds',
                           color='r',
                           label='Model',
                           **plt_kwargs)

        # Target scatter plot
        dfp = self._check_type(target)
        dfp.sort_values(self._xcol, inplace=True)
        plt_kwargs = self._get_cfg_kwargs(['s'])
        dfp.plot.scatter(self._xcol,
                         'seconds',
                         label='Executions',
                         ax=self.ax,
                         alpha=0.4,
                         **plt_kwargs)
        # Plot target median values per worker amount
        medians = dfp[[self._xcol, 'seconds']].groupby(self._xcol).median(
            ).rename(columns={'seconds': 'Median'})
        medians.plot(style='b--', ax=self.ax)

        self._finalize(dfp)
        if output is not None:
            plt.savefig(output)
        plt.show()

    def plot_cost_model(self, model, target, cost_func, output=None):
        """Cost prediction.

        Args:
            cost_func (function): return money cost of 1 VM given duration in
                seconds.
        """
        # # Target # #
        # Target scatter plot
        tgt_df = target[['workers', 'duration_ms']].groupby(
            'workers').median()
        tgt_df['seconds'] = tgt_df.duration_ms / 1000
        tgt_df['money'] = tgt_df.seconds.apply(cost_func) * (tgt_df.index + 1)

        plt_kwargs = self._get_cfg_kwargs(['s'])
        self.ax = tgt_df.plot.scatter('money', 'seconds', alpha=0.4,
                                      **plt_kwargs)

        tgt_df.sort_values('money', inplace=True)
        tgt_df.plot('money', 'seconds', style='b--', label='Median',
                    ax=self.ax)

        # # Prediction # #
        # Generate an X matrix with more workers to smooth model line.
        df = _add_x(target)

        # Adding model prediction column
        df['seconds'] = model.predict(df) / 1000
        # +1 for master node
        df['money'] = df.seconds.apply(cost_func) * (df.workers + 1)

        # Plot prediction values
        plt_kwargs = self._get_cfg_kwargs(['figsize', 'logx', 'logy'])
        df.sort_values('money', inplace=True)
        self.ax = df.plot('money', 'seconds', color='r', label='Model',
                          ax=self.ax, **plt_kwargs)

        # # Labels, etc. # #
        xlim, ylim, loc = self._get_cfg_args(['xlim', 'ylim', 'loc'])
        if xlim:
            self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('cost (USD)')
        self.ax.set_ylabel('seconds')
        if loc:
            plt.legend(loc=loc)
        plt.tight_layout()

        if output is not None:
            plt.savefig(output)
        plt.show()

    def _check_type(self, df):
        dfp = df.copy()
        if 'workers' in dfp.columns and dfp.workers.unique().size > 1:
            self._xcol = 'workers'
            self._labels = ['Executions', 'Outliers (> 1.5 * IQR)',
                            'Non-outlier mean']
            self._xlabel = 'workers'
            dfp.workers = dfp.workers.astype('int')
        elif dfp.input.unique().size > 1:
            self._xcol = 'input'
            self._labels = ['Executions', 'Outliers (> 1.5 * IQR)', 'Mean']
            self._xlabel = 'input size (MiB)'
            # From bytes to MiB
            dfp.input = (dfp.input / 1024**2).round().astype('int')
        else:
            print('workers not in', dfp.columns)
            print('only one size', dfp.input.unique())
            print(dfp.head())

        # From milliseconds to seconds
        dfp.duration_ms /= 1000
        dfp.rename(columns={'duration_ms': 'seconds'}, inplace=True)

        return dfp

    def _plot_outliers(self, outliers, safe):
        kwargs = self._get_cfg_kwargs(['s', 'figsize', 'logx', 'logy'])
        # Non-outlier scatter plot
        self.ax = safe.plot.scatter(self._xcol,
                                    'seconds',
                                    label=self._labels[0],
                                    zorder=2,
                                    **kwargs)
        if not outliers.empty:
            # Outlier scatter plot
            outliers.plot.scatter(self._xcol,
                                  'seconds',
                                  label=self._labels[1],
                                  color='r',
                                  ax=self.ax,
                                  zorder=1,
                                  s=self._cfg['s'])
        # Mean (non-outliers)
        means = safe[[self._xcol, 'seconds']].groupby(self._xcol).mean(
        ).rename(columns={'seconds': self._labels[2]})
        means.plot(color='b', ax=self.ax, zorder=3)

    def _finalize(self, df):
        plt.xticks(df[self._xcol].unique(), df[self._xcol].unique())
        xlim, ylim, loc = self._get_cfg_args(['xlim', 'ylim', 'loc'])
        if xlim:
            self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        if self._cfg['logx']:
            self.ax.set_xlabel(self._xlabel + ' (log)')
        else:
            self.ax.set_xlabel(self._xlabel)
        if self._cfg['logy']:
            self.ax.set_ylabel('seconds (log)')
        if loc:
            plt.legend(loc=loc)
        plt.tight_layout()

    def _get_cfg_kwargs(self, keys):
        return {k: self._cfg[k] for k in keys}

    def _get_cfg_args(self, keys):
        return [self._cfg[k] for k in keys]


def _add_x(target):
    """Generate an X matrix with more workers to smooth model line."""
    # Only one input size
    input_size = target.input.unique()
    assert len(input_size) == 1

    # Plot every integer between min and max workers
    min_w = target.workers.min()
    max_w = target.workers.max()
    num = max_w - min_w + 1
    workers = np.linspace(min_w, max_w, num)

    # DataFrame with all workers and the same input size
    df = pd.DataFrame({'workers': workers, 'input': input_size[0]})

    # Add all features except duration_ms and its log
    creator = FeatureCreator(df)
    cols = df.columns.tolist()
    ycols = [Config.Y, Config.LOG_Y]
    extra_features = [col for col in Config.EXTRA_FEATURES if col[0] not in
                      ycols]
    no_expansion = [col for col in Config.NO_EXPANSION if col in cols]
    creator.add_cols(extra_features)
    return creator.expand_poly(Config.DEGREE, no_expansion,
                               Config.DEL_FEATURES)

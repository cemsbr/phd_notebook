"""The one and only module for the sake of documentation."""
import matplotlib.pyplot as plt


class Plotter:
    """Plotting helper"""

    def __init__(self,
                 xlim=None,
                 ylim=(0, None),
                 figsize=(8, 5),
                 loc=None,
                 logx=False,
                 logy=False):
        self._cfg = {'xlim': xlim,
                     'ylim': ylim,
                     'figsize': figsize,
                     'loc': loc,
                     'logx': logx,
                     'logy': logy,
                     's': 80}
        self.ax = None
        self._xcol = None
        self._xlabel = None
        self._labels = None

    def plot_outliers(self, df):
        """Plot total durations by number of workers."""
        dfp = self._check_type(df)
        self._plot_outliers(dfp[df.outlier], dfp[~df.outlier])
        self._finalize(dfp)
        plt.show()

    def plot_model(self, model, target):
        """Compare model and target experiment.

        :param model: lib.Model
        :param target: DataFrame of target experiment (workers, input, ms),
            without outliers
        """
        # Prediction features DataFrame with unique values of workers and
        # input size
        pred_df = target.drop('ms',
                              axis=1).drop_duplicates().sort_values('workers')
        # Adding model prediction column
        pred_df['Model'] = model.predict(pred_df) / 1000
        pred_df.drop('input', axis=1, inplace=True)

        # Plot prediction values
        plt_kwargs = self._get_cfg_kwargs(['figsize', 'logx', 'logy'])
        self.ax = pred_df.plot('workers', 'Model', color='r', **plt_kwargs)

        # Target scatter plot
        dfp = self._check_type(target)
        plt_kwargs = self._get_cfg_kwargs(['s'])
        dfp.plot.scatter('workers', 'seconds', label='Executions', ax=self.ax, **plt_kwargs)
        # Plot target mean values per worker amount
        means = dfp[['workers', 'seconds']].groupby('workers').mean().rename(
            columns={'seconds': 'Execution mean'})
        means.plot(style='b--', ax=self.ax)

        self._finalize(target)
        plt.show()

    def _check_type(self, df):
        dfp = df.copy()
        if df.workers.unique().size > 1:
            self._xcol = 'workers'
            # From milliseconds to seconds
            self._labels = ['Executions', 'Outliers (> 1.5 * IQR)',
                            'Non-outlier mean']
            self._xlabel = 'workers'
        elif df.input.unique().size > 1:
            self._xcol = 'input'
            # From bytes to MiB, from milliseconds to seconds
            self._labels = ['Executions', 'Outliers (> 1.5 * IQR)', 'Mean']
            self._xlabel = 'input size (MiB)'

        dfp.ms /= 1000
        dfp.input = (df.input / 1024**2).round().astype('int')
        dfp.rename(columns={'ms': 'seconds'}, inplace=True)

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

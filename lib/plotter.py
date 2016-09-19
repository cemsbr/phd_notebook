"""The one and only module for the sake of documentation."""
import matplotlib.pyplot as plt


class Plotter:
    """Plotting helper."""

    def __init__(self,
                 xlim=None,
                 ylim=(0, None),
                 figsize=(4.5, 3),
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

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            self._cfg[k] = v

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
        target = target.copy()

        pred_df = target.drop(model.y, axis=1)
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
                         **plt_kwargs)
        # Plot target median values per worker amount
        medians = dfp[[self._xcol, 'seconds']].groupby(self._xcol).median(
            ).rename(columns={'seconds': 'Median'})
        medians.plot(style='b--', ax=self.ax)

        self._finalize(dfp)
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

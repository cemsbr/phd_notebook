"""The one and only module for the sake of documentation."""
import matplotlib.pyplot as plt
from pandas import DataFrame


class Plotter:
    """Plotting helper"""

    def __init__(self):
        self.ax = None
        self._cfg = None
        self._xcol = None
        self._xlabel = None
        self._labels = None

    def plot_outliers(self,
                      df,
                      xlim=None,
                      ylim=(0, None),
                      figsize=(8, 5),
                      loc=None,
                      logx=False,
                      logy=False):
        """Plot total durations by number of workers."""
        self._cfg = {'xlim': xlim,
                     'ylim': ylim,
                     'figsize': figsize,
                     'loc': loc,
                     'logx': logx,
                     'logy': logy,
                     's': 80}

        dfp = self._set_type(df)
        self._plot(dfp[df.outlier], dfp[~df.outlier])
        self._config_plot(dfp)
        plt.show()

    def _set_type(self, df):
        if df.workers.unique().size > 1:
            self._xcol = 'workers'
            # From milliseconds to seconds
            dfp = DataFrame({'workers': df.workers, 'seconds': df.ms / 1000})
            self._labels = ['Executions', 'Outliers (> 1.5 * IQR)',
                            'Non-outlier mean']
            self._xlabel = 'workers'
        elif df.input.unique().size > 1:
            self._xcol = 'input'
            # From bytes to MiB, from milliseconds to seconds
            dfp = DataFrame({'input': (df.input / 1024**2).round().astype(
                'int'),
                             'seconds': df.ms / 1000})
            self._labels = ['Executions', 'Outliers (> 1.5 * IQR)', 'Mean']
            self._xlabel = 'input size (MiB)'

        return dfp

    def _plot(self, outliers, safe):
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
        means = safe.groupby(self._xcol).mean().rename(
            columns={'seconds': self._labels[2]})
        means.plot(color='b', ax=self.ax, zorder=3)

    def _config_plot(self, df):
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

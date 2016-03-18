# pylint: disable=invalid-name, missing-docstring
import matplotlib.pyplot as plt
import numpy as np
from lib import R2


class Plotter:

    #
    # BEGIN refactoring
    #

    def __init__(self, ylabel, scale=None, size=(12, 5)):
        self.ylabel = ylabel
        self._xticks = set()
        if scale is None:
            scale = 1
        self._scale_ys = lambda ys: np.array(list(ys)) * scale
        plt.subplots(figsize=size)
        self._xticks_set = False

    def scatter(self, xs, ys, *args, **kwargs):
        ys_scaled = self._scale_ys(ys)
        plt.scatter(list(xs), ys_scaled, *args, s=40, alpha=0.2, **kwargs)

    def plot_means(self, xs, y_lists, *args, **kwargs):
        means = (np.mean(list(values)) for values in y_lists)
        ys = self._scale_ys(means)
        if 'label' not in kwargs:
            kwargs['label'] = 'Means'
        xs_list = list(xs)
        self.add_xticks(xs_list)
        plt.plot(xs_list[:len(ys)], ys, *args, **kwargs)

    def plot_polynomial(self, xs, poly, *args, **kwargs):
        full_xs = get_linspace(xs)
        ys = self._scale_ys(np.polyval(poly, full_xs))

        if 'label' not in kwargs:
            kwargs['label'] = 'Regression'
        if 'r2' in kwargs:
            kwargs['label'] += ' ({})'.format(get_r2_label(kwargs['r2']))
            del kwargs['r2']

        plt.plot(full_xs, ys, *args, **kwargs)

    def plot_model(self, xs, model, r2=None, *args, **kwargs):
        full_xs = get_linspace(xs)
        ys = self._scale_ys(model(x) for x in full_xs)

        if 'label' not in kwargs:
            kwargs['label'] = 'Model'
        if r2 is not None:
            kwargs['label'] += ' ({})'.format(get_r2_label(r2))

        plt.plot(full_xs, ys, *args, **kwargs)

    def add_xticks(self, xs):
        xs_list = list(xs)
        self._xticks.update(xs_list)

    def set_xticks(self, location, labels=None):
        plt.xticks(location, labels)
        self._xticks_set = True

    def end(self, loc=0, ymin=0, ymax=None):
        if not self._xticks_set:
            xs = sorted(list(self._xticks))
            set_xlim(xs)
            plt.xticks(xs)
        plt.xlabel('Nodes')
        plt.ylabel(self.ylabel)
        plt.ylim(ymin, ymax)
        leg = plt.legend(loc=loc)
        leg.get_frame().set_alpha(0.5)
        plt.show()

    #
    # END refactoring
    #

    def plot_stage_durations(self, xp, stage):
        xs = xp.n_nodes
        y_lists = [list(stages) for stages in xp.get_stage_lists(stage)]
        self._start_stage_plot()
        self.plot_means(xs, y_lists)
        self.match_and_scatter(xs, y_lists, alpha=0.2, s=40, label='Stage')
        Plotter._finish(xs)

    def plot_stage_means(self, xp, stage, *args, **kwargs):
        self._start_stage_plot()
        self.plot_means(xp.n_nodes, xp.get_stage_lists(stage), *args, **kwargs)
        Plotter._finish(xp.n_nodes)

    def plot_stage_model(self, xp, stage, model, label):
        self._start_stage_plot()
        self.plot_means(xp.n_nodes, xp.get_stage_lists(stage), '-o')
        label_r2 = self._get_stage_r2_label(xp, stage, model, label)
        self._plot_model(xp.n_nodes, model, 'r-', label=label_r2)
        Plotter._finish(xp.n_nodes)

    def _get_stage_r2_label(self, xp, stage, model, prefix):
        xs = (a.workers for a in xp.get_apps())
        ys = (s.duration for s in xp.get_stage(stage))
        return Plotter._get_r2_label(xs, ys, model, prefix)

    @staticmethod
    def _get_r2_label(xs, ys, model, prefix):
        r2c = R2(xs, ys)
        r2c.calc_from_fn(model)
        return r2c.get_label(prefix)

    def _plot_model(self, xs, model, *args, **kwargs):
        xi, xf = xs[0], xs[-1]
        xs_full = np.linspace(xi, xf, xf - xi + 1)
        ys = self._scale_ys(model(x) for x in xs_full)
        plt.plot(xs_full, ys, *args, **kwargs)
        #ys = self._scale_ys(model(x) for x in xs)
        #plt.plot(xs, ys, '-o')

    def _start_stage_plot(self):
        self.set_labels('VMs', 'Duration (sec)')

    @staticmethod
    def _finish(xs):
        plt.xticks(xs)
        plt.xlim(xs[0] - 2, xs[-1] + 2)
        plt.ylim(ymin=0)
        plt.legend(loc=0)

    def _plot_stage_means(self, xs, y_lists):
        self.plot_means(xs, y_lists, label='VM-quantity mean')

    def plot_means_back(self, xs, timed_lists, *args, **kwargs):
        ys = (np.mean([t.duration for t in timeds]) for timeds in timed_lists)
        ys = self._scale_ys(ys)
        plt.plot(xs[:len(ys)], ys, label='VM-quantity mean', *args, **kwargs)

    def scatter_back(self, xs, timeds, *args, **kwargs):
        """Make a scatter plot.
        :param iter timeds: Iterable of Timed objects
        """
        ys = (t.duration for t in timeds)
        ys = self._scale_ys(ys)
        plt.scatter(xs, ys, *args, **kwargs)
        plt.xticks(list(set(xs)))

    def match_and_scatter(self, xs, timed_lists, *args, **kwargs):
        """Match dimensions of xs and ys and scatter.
        :param iter xs: dimensions of each timed list
        :param iter ys: lists of timed objects
        """
        all_xs, ys = [], []
        for x, timeds in zip(xs, timed_lists):
            all_xs.extend([x] * len(timeds))
            ys.extend(timeds)
        self.scatter(all_xs, ys, *args, **kwargs)
        return all_xs

    @staticmethod
    def set_labels(xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


def get_r2_label(r2):
    return r'$R^{2}$ = ' + '{:.4f}'.format(r2)

def get_linspace(xs):
    x0, xf = xs[0], xs[-1]
    return np.linspace(x0, xf, xf - x0 + 1)

def set_xlim(xs):
    x0, xf = xs[0], xs[-1]
    offset = (xf - x0)/20
    plt.xlim(x0 - offset, xf + offset)

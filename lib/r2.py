#pylint: disable=missing-docstring, invalid-name
import numpy as np


class R2:
    def __init__(self, xs, ys):
        """
        :param iter xs: expected xs
        :param iter ys: expected ys
        """
        self._xs = xs
        self._ys = np.array(list(ys))
        self._r2 = None

        ys_mean = self._ys.mean()
        self._sst = sum(((y - ys_mean) ** 2 for y in self._ys))

    # r-squared calculation
    def calc(self, model_ys):
        sse = sum(((y - y_) ** 2 for y, y_ in zip(self._ys, model_ys)))
        self._r2 = (self._sst - sse) / self._sst
        return self._r2

    def calc_from_fn(self, fn):
        return self.calc(fn(x) for x in self._xs)

    def get_label(self, prefix):
        return '{}, '.format(prefix) + r'$R^{2}$ = ' + \
               '{:.4f}'.format(self._r2)

"""The one and only module for the sake of documentation."""
class Humanizer:
    """Make it more readable by humans."""

    SIZE_GIB = ('input size (GiB)', 1024**3)
    SIZE_MIB = ('input size (MiB)', 1024**2)
    TIME_SEC = ('seconds', 10**3)

    def __init__(self, size=None, time=None):
        if size is None:
            size = Humanizer.SIZE_GIB
        if time is None:
            time = Humanizer.TIME_SEC
        self._size = size
        self._time = time

    def humanize(self, df):
        # Avoid changing indexed values (don't know how)
        dfh = df.reset_index()

        _replace(dfh, 'input', self._size[0], self._size[1])
        _replace(dfh, 'ms', self._time[0], self._time[1])

        return dfh


def _replace(df, col, label, div):
    if col in df.columns:
        df[col] = (df[col] / div).round().astype(int)
        df.rename(columns={col: label}, inplace=True)

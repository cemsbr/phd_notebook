"""The one and only module for the sake of documentation."""
class Humanizer:
    SIZE_GIB = ('GiB', 1024**3)
    SIZE_MIB = ('MiB', 1024**2)
    TIME_SEC = ('sec', 10**3)

    @staticmethod
    def humanize(df, size=None, time=None):
        if size is None:
            size = Humanizer.SIZE_GIB
        if time is None:
            time = Humanizer.TIME_SEC

        # Avoid changing indexed values
        dfh = df.reset_index()

        _replace(dfh, 'input', size[0], size[1])
        _replace(dfh, 'ms', time[0], time[1])

        return dfh


def _replace(df, col, label, div):
    if col in df.columns:
        df[col] = (df[col] / div).round().astype(int)
        df.rename(columns={col: label}, inplace=True)

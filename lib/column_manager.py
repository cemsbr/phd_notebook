"""Manage DataFrame columns."""
import numpy as np


class ColumnManager:
    """Create column sets with current and new columns.

    Column sets and y columns are class attributes. The source DataFrame is an
    object attribute.
    """

    #: key = name, value = (cols, new_cols, is_log)
    _col_sets = {}
    _y_col = None

    def __init__(self):
        """Initialize object attributes."""
        #: pd.DataFrame: source DataFrame.
        self._src = None
        #: pd.DataFrame: target DataFrame.
        self._dst = None

    @classmethod
    def add_col_set(cls, name, new_cols=None, cols=None, is_log=False):
        """Add a column set.

        Args:
            name (str): Column set name.
            cols (str): Column names from source DataFrame.
            new_cols (tuple list): Functions that return column name and
                values. They will receive the source DataFrame.
            is_log (boolean): Whether to apply log in the prediction target
                column (:meth:`set_y_column`).
        """
        if cols is None:
            cols = []
        if new_cols is None:
            new_cols = []
        cls._col_sets[name] = (cols, new_cols, is_log)

    @classmethod
    def set_y_column(cls, y_col):
        """Set prediction target column names.

        Features will be all columns but *y_col*.

        Args:
            y_col (str): Column name of the generated DataFrame.
        """
        cls._y_col = y_col

    def set_source_df(self, df):
        """Set source DataFrame.

        Args:
            src (pd.DataFrame): Source DataFrame.
        """
        self._src = df
        # Clear the destination DataFrame.
        self._dst = None

    def get_col_sets(self):
        """Return a generator of column set name, x and y."""
        for col_set in sorted(self._col_sets):
            x, y = self.get_xy(col_set)
            is_log = self._col_sets[col_set][2]
            yield col_set, x, y, is_log

    def is_log(self, col_set):
        """Return whether the column set predicts *log(y)* instead of *y*."""
        return self._col_sets[col_set][2]

    def get_xy(self, col_set):
        """Return both feature and target prediction DataFrames."""
        return self.get_x(col_set), self.get_y(col_set)

    def get_x(self, col_set):
        """Return column set feature DataFrame."""
        return self._get_dst(col_set)

    def get_y(self, col_set):
        """Return column set y (prediction target).

        If column set has *is_log* as *True*, return y's log
        """
        y = self._src[self._y_col]
        return np.log(y) if self.is_log(col_set) else y

    def _get_dst(self, col_set):
        cols, new_cols, _ = self._col_sets[col_set]
        df = self._src[cols].copy()
        for fn in new_cols:
            col, values = fn(self._src)
            df[col] = values
        return df

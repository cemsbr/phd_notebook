"""Provides :class:`StageDataFrame` to extract stage columns."""
import re

from lib.group_outlier import GroupOutlier


class StageDataFrame:
    """Extract columns of a stage from a DataFrame."""

    _STAGE_Y = 's_dur'

    def __init__(self, src):
        """Extract *stage* cols from *df*.

        Args:
            src (pd.DataFrame): Source DataFrame (e.g. application DF).
            y_cols (list): Column names of prediction target.
            stage (int): Stage number.
        """
        self._src = src
        self._stages, self._non_stage_cols = self._get_stages()
        #: pd.DataFrame: Stage DataFrame, with column names without stage
        #: prefix.
        self._df = None

    def get_stages_df(self, remove_outliers):
        """Iterate over stage DataFrames.

        A stage DataFrame contains stage columns without stage prefix (e.g.
        *tasks* instead of *s01_tasks*) and other columns from source that are
        not related to stages, that doesn't have the stage prefix originally
        (e.g. *workers*).

        Returns:
            int: Stage number.
            pd.DataFrame: Stage DataFrame.
        """
        for stage in self._stages:
            yield stage, self.get_stage_df(stage, remove_outliers)

    def _get_stages(self):
        """Return all stages of source DF, as integers, and non-stage cols."""
        p = re.compile(r'^s(\d+)_')
        stages = set()
        non_stages = []
        rows = len(self._src)
        # Remove cols having all rows empty
        for col in self._src.dropna(axis=1, thresh=rows).columns:
            match = p.match(col)
            if match:
                stage_nr = int(match.group(1))
                stages.add(stage_nr)
            else:
                non_stages.append(col)
        return sorted(stages), non_stages

    def get_stage_df(self, stage, remove_outliers):
        """Rename stage columns and remove outliers."""
        df = self._get_stage_all(stage)
        if remove_outliers:
            df = GroupOutlier.remove_outliers(df, self._STAGE_Y)
        return df

    def _get_stage_all(self, stage):
        prefix = 's{:02d}_'.format(stage)
        new_names = self._get_new_names(prefix)
        cols = self._non_stage_cols + sorted(new_names)
        return self._src[cols].rename(columns=new_names)

    def _get_new_names(self, prefix):
        """Remove stage number, so user doesn't have to deal with it."""
        suff_start = len(prefix)
        cols = self._src.columns
        return {c: 's_' + c[suff_start:] for c in cols if c.startswith(prefix)}

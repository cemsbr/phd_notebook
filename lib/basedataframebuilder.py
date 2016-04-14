"""The one and only module for the sake of documentation."""
import pandas as pd


class BaseDataFrameBuilder:
    """Help building data frames from Spark log files."""

    def __init__(self, threads, stage):
        self._threads = threads
        self._stage = stage

    @staticmethod
    def from_worker_input(pairs_list):
        """Workers and input DataFrame from a list of value pairs."""
        return pd.DataFrame.from_records(pairs_list,
                                         columns=['workers', 'input'])

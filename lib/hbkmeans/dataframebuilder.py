from lib import BaseDataFrameBuilder


class DataFrameBuilder(BaseDataFrameBuilder):
    _FOLDER = 'data/hibench/kmeans'
    _INPUT_THRESHOLD = 8 * 10**6

    def __init__(self, threads=None, stage=None):
        super().__init__(threads, stage)
        self._df = self._build_all(DataFrameBuilder._FOLDER)

    def get_profiling(self):
        return self._df[self._df.input < DataFrameBuilder._INPUT_THRESHOLD]

    def get_target(self):
        return self._df[self._df.input > DataFrameBuilder._INPUT_THRESHOLD]

    def free(self):
        self._df = None

    def _get_records(self, apps):
        records = []
        for app in apps:
            records.append((
                app.stages[0].records_read,
                app.slaves,
                self._get_duration(app)
            ))
        columns = ('input', 'workers', 'ms')
        return records, columns

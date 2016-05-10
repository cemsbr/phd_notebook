from lib import BaseDataFrameBuilder


class DataFrameBuilder(BaseDataFrameBuilder):
    _FOLDER = 'data/hibench/kmeans'
    _INPUT_THRESHOLD = 8 * 10**6

    def __init__(self, threads=None, stage=None):
        super().__init__(threads, stage)
        self._df = self._build_all(DataFrameBuilder._FOLDER)
        _input2samples(self._df)

    def get_profiling(self):
        return self._df[self._df.input < DataFrameBuilder._INPUT_THRESHOLD]

    def get_target(self):
        return self._df[self._df.input > DataFrameBuilder._INPUT_THRESHOLD]

    def free(self):
        self._df = None


# From input size, infer number of samples
def _input2samples(df):
    b2s = {}
    k_samples = 32000
    for byetes in sorted(df.input.unique()):
        b2s[byetes] = k_samples
        k_samples *= 2

    df.input = df.input.apply(lambda b: b2s[b])

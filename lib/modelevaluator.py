import pickle
from multiprocessing import Pool
from functools import partial

class ModelEvaluator:
    def __init__(self, model_creator=None):
        self._creator = model_creator
        self._fit_df = None

    def set_fit_data(self, df):
        self._fit_df = df

    def evaluate(self, dfs, is_log=False):
        """Fit once, evaluate many times.

        Return an evaluation list for every df in dfs.
        """
        eval_model = partial(_eval_model, self._fit_df, dfs, is_log)
        with Pool() as p:
            res = p.map(eval_model, self._creator.get_models())
        return list(zip(*res))

    def evaluate_model(self, dfs, model):
        model.fit(self._fit_df)
        return [_eval_df(model, df) for df in dfs]


def _eval_model(fit_df, dfs, is_log, model):
    model.fit(fit_df)
    model.is_log = is_log
    return [_eval_df(model, df) for df in dfs]


def _eval_df(model, df):
    return {
        'scores': model.score(df),
        'model_dump': pickle.dumps(model)
    }

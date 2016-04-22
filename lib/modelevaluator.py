class ModelEvaluator:
    def __init__(self, model_creator, ycol):
        """
        :param iterable models: Trained models.
        """
        self._model_creator = model_creator
        self._ycol = ycol
        self._fit_x, self._fit_y = None, None
        self._score_x, self._score_y = None, None

    def fit(self, fit_df):
        self._fit_x, self._fit_y = self._split_xy(fit_df)

    def evaluate(self, df):
        self._score_x, self._score_y = self._split_xy(df)
        models = self._model_creator.get_models()
        return [(str(model), self._evaluate(model)) for model in models]

    def _evaluate(self, model):
        model.fit(self._fit_x, self._fit_y)
        return model.score(self._score_x, self._score_y)

    def _split_xy(self, df):
        x = df.drop(self._ycol, axis=1)
        y = df[self._ycol].copy()
        return (x, y)

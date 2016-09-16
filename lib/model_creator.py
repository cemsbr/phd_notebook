"""Feature sets generator."""
from itertools import chain, combinations


class ModelCreator:
    """Generate models to be evaluated."""

    def __init__(self, linear_models, features, y, log_features, log_y):
        """Linear models and features to generate models.

        Args:
            linear_models (list): Linear models, e.g. `Ridge`.
            features (list): Features to generate a powerset to predict
                `y`.
            y (str): column to be predicted.
            log_features (list): Features to generate a powerset to
                predict `log(y)``.
            log_y (str): column whose log is to be predicted.
        """
        self.linear_models = linear_models
        self.features = features
        self.y = y
        self.log_features = log_features
        self.log_y = log_y

    def get_models(self, cpu, cpus):
        """Divide models into `cpus` groups.

        Args:
            cpu (int): CPU number starting from 0.
            cpus (int): total number of CPUs.

        Returns:
            generator: list of all models divided into `cpus` lists.
        """
        # Count processed models by previous CPUs.
        remaining = len(self)
        for i in range(cpu):
            remaining_cpus = cpus - i
            cpu_models = round(remaining / remaining_cpus)
            remaining -= cpu_models
        processed = len(self) - remaining

        remaining_cpus = cpus - cpu
        cpu_models = round(remaining / remaining_cpus)
        return self._get_models(processed, cpu_models)

    def __len__(self):
        """Number of models."""
        total = 2**len(self.features) - 1
        total += 2**len(self.log_features) - 1
        total *= len(self.linear_models)
        return total

    def _get_models(self, skip, length):
        """Generate models.

        Args:
            skip (int): Models to skip in the beginning.
            length (int): How many models to return.

        Returns:
            generator: Models.
        """
        i = 0
        for linear_model in self.linear_models:
            for features in _powerset(self.features):
                if i < skip:
                    i += 1
                elif i == skip + length:
                    return
                else:
                    i += 1
                    yield Model(i, linear_model, list(features), self.y, False)
            for features in _powerset(self.log_features):
                if i < skip:
                    i += 1
                elif i == skip + length:
                    return
                else:
                    i += 1
                    yield Model(i, linear_model, list(features), self.log_y,
                                True)


class Model:
    """Prediction model."""

    def __init__(self, number=None, linear_model=None, features=None, y=None,
                 is_log=None):
        """Define the model.

        Args:
            number (int): Model number.
            linear_model: For example, `RidgeCV`.
            features (list): List of column names.
            y (str): Column to be predicted.
            is_log (boolean): Whether we are predicting the duration log.
        """
        self.number = number
        self.linear_model = linear_model
        self.features = features
        self.y = y
        self.is_log = is_log

    def humanize(self):
        """For humans to know about the model."""
        human = {'log': self.is_log}
        params = {}

        all_params = self.linear_model.get_params()
        cls = type(self.linear_model).__name__
        if cls == 'LinearRegression':
            human['linear model'] = 'LinReg'
        elif cls == 'RidgeCV':
            human['linear model'] = cls
            params['alphas'] = all_params['alphas']
            # try:
            params['best'] = self.linear_model.alpha_
            # except AttributeError:
            #    params['best'] = None
        else:
            human['linear model'] = cls
            params.update(all_params)

        human['features'] = self.features
        human['number of features'] = len(self.features)
        human['number'] = self.number
        human['params'] = ', '.join('{}: {}'.format(k, params[k])
                                    for k in sorted(params))
        return human

    def __str__(self):
        """Multiple-line output."""
        h = self.humanize()
        return '      model number: {}\n' \
            '      linear model: {}\n' \
            '      duration log: {}\n' \
            '            params: {}\n' \
            'number of features: {}\n' \
            '          features: {}'.format(h['number'], h['model'], h['log'],
                                            h['params'], h['nr_feats'],
                                            ', '.join(h['features']))


def _powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r + 1) for r in range(len(s)))


def _get_next(generator, n):
    res = []
    try:
        for _ in range(n):
            res.append(next(generator))
    except StopIteration:
        pass
    return res

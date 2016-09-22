"""Configuration."""
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV


class Config:
    """Configuration."""

    #: Extra features and how to create them.
    EXTRA_FEATURES = (
        ('1/workers', lambda df: 1/df.workers),
        ('log(input/workers)', lambda df: np.log2(df.input/df.workers)),
        ('log(input)', lambda df: np.log2(df.input)),
        ('log(workers)', lambda df: np.log2(df.workers)),
        ('log(duration_ms)', lambda df: np.log2(df.duration_ms)),
    )

    #: Features for log regression.
    LOG_FEATURES = ['log(input)', 'log(workers)']
    #: What we are trying to predict in log regression.
    LOG_Y = 'log(duration_ms)'

    #: Features for regular regression.
    FEATURES = ['input', 'workers', '1/workers', 'log(input/workers)',
                'log(input)', 'log(workers)']
    #: What we are trying to predict in regular regression.
    Y = 'duration_ms'

    #: Polynomial expansion degree.
    DEGREE = 2
    #: Columns that should not be expanded.
    NO_EXPANSION = ['application', 'set', 'duration_ms', 'log(duration_ms)']

    #: linear models
    LINEAR_MODELS = (LinearRegression(),
                     RidgeCV(normalize=True, alphas=(0.01, 0.1, 1, 3, 10)))

    #: Features to remove because of long time required for processing.
    #: Factors must be sorted.
    DEL_FEATURES = [
        'input * log(input/workers)',
        'input * log(input)',
        # 'input * log(workers)',
        '1/workers * workers',
        'log(input/workers) * workers',
        # 'log(input) * workers',
        'log(workers) * workers',
        '1/workers * log(input/workers)',
        # '1/workers * log(input)',
        '1/workers * log(workers)',
        'log(input) * log(input/workers)',
        'log(input/workers) * log(workers)',
    ]

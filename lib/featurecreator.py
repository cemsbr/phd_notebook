from itertools import combinations_with_replacement as get_combs, chain
import numpy as np
import pandas as pd


class FeatureCreator:
    def __init__(self, df):
        self._df = df

    def get_poly(self, degree):
        df = pd.DataFrame()
        all_powers = self._get_all_powers(degree)
        for powers in all_powers:
            col = self._get_new_col_name(powers)
            value = self._get_new_col_value(powers)
            df[col] = value
        _drop_infnan(df)
        return df

    def get_log(self):
        df = pd.DataFrame()
        for col, value in self._df.items():
            df['log({})'.format(col)] = np.log2(value)
        _drop_infnan(df)
        return df

    def _get_all_powers(self, degree):
        n_features = self._df.shape[1]
        powers = chain.from_iterable(get_combs(
            range(n_features), d) for d in range(1, degree + 1))
        return (np.bincount(p, minlength=n_features) for p in powers)

    def _get_new_col_value(self, powers):
        value = np.ones(len(self._df))
        for col, power in zip(self._df.columns, powers):
            if power > 0:
                value *= self._df[col]**power
        return value

    def _get_new_col_name(self, powers):
        cols = []
        for df_col, power in zip(self._df.columns, powers):
            if power == 0:
                continue
            col = df_col
            if power > 1:
                col = '({})^{:d}'.format(col, power)
            cols.append(col)
        return ' * '.join(cols)


def _drop_infnan(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

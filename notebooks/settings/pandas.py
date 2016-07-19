import pandas as pd


pd.options.display.precision = 2
pd.options.display.max_columns = 99
pd.options.display.max_rows = 99
pd.options.display.float_format = lambda x: '{:.2f}'.format(x)

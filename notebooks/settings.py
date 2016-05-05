import matplotlib
import pandas as pd


matplotlib.style.use('ggplot')  # R style
matplotlib.rcParams.update({'font.size': 16,
                            'font.family': 'serif',
                            'lines.linewidth': 2,
                            'axes.xmargin': 0.1,
                            'axes.ymargin': 0.1})
pd.options.display.precision = 2
pd.options.display.max_columns = 99
pd.options.display.float_format = lambda x: '{:.2f}'.format(x)

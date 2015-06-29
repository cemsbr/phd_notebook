# Notebook configuration and modules import

import glob

from IPython.core.display import Markdown

# Show graph in this notebook instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sparklogstats import LogParser


### Parsing ###

# Parse and sort Spark applications' log by execution start time
def parse(files):
    apps = []
    worker_amounts = set()
    parser = LogParser()
    for log in glob.glob(files):
        parser.parse_file(log)
        app = parser.app
        app.workers = len(app.workers)  # not using hostnames for now
        apps.append(app)
        worker_amounts.add(app.workers)
    apps.sort(key=lambda x: x.start)
    
    return apps, sorted(worker_amounts)

def parse_strong_scaling():
    apps, worker_amounts = parse('strong_scaling_data/app-*')
    # Removing one outlier: workers = 8 and duration > 200s
    # apps = (a for a in apps if not (a.workers == 8 and a.duration > 200000))

    # immutable to prevent mistakes
    return tuple(apps), tuple(worker_amounts)

def parse_weak_scaling(workers=None):
    apps, worker_amounts = parse('weak_scaling_data/app-*')
    # immutable to prevent mistakes
    if workers is not None:
        apps = tuple((a for a in apps if a.workers in workers))
    return apps, tuple(worker_amounts)

### Some convenient functions ###

def plt_labels(xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def get_r2_label(prefix, r2):
    return '{}, '.format(prefix) \
        + r'$R^{2}$ = ' \
        + '{:.4f}'.format(r2)
        
# r-squared calculation
def calc_r2(xs, ys, model_ys):
    ys_mean = np.mean(ys)
    sst = sum(((y - ys_mean)**2 for y in ys))
    sse = sum(((y - y_)**2 for y, y_ in zip(ys, model_ys)))
    return (sst - sse)/sst

def calc_r2_f(xs, ys, f):
    model_ys = (f(x) for x in xs)
    return calc_r2(xs, ys, model_ys)
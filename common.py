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
def parse(files, workers=None):
    apps = []
    worker_amounts = set()
    parser = LogParser()
    for log in glob.glob(files):
        parser.parse_file(log)
        app = parser.app
        app.workers = len(app.workers)  # not using hostnames for now
        if workers is not None and app.workers not in workers:
            continue
        apps.append(app)
        worker_amounts.add(app.workers)
    apps.sort(key=lambda app: app.start)
    apps.sort(key=lambda app: app.workers)
    
    return apps, sorted(worker_amounts)

def parse_strong_scaling(map_outliers=False, reduce_outliers=False):
    apps, worker_amounts = parse('strong_scaling_data/app-*')
    if not map_outliers:
        # Removing one map outlier
        apps = (a for a in apps if not (a.workers == 8 and a.duration > 200000))
    if not reduce_outliers:
        # Removing 5 reduce outliers
        apps = (a for a in apps if not (a.workers == 64 and a.stages[1].duration > 15000))

    # immutable to prevent mistakes
    return tuple(apps), tuple(worker_amounts)

def parse_weak_scaling(workers=None):
    apps, worker_amounts = parse('weak_scaling_data/app-*', workers)
    # immutable to prevent mistakes
    return tuple(apps), tuple(worker_amounts)

# sss = small strong scaling
def parse_sss(workers=None):
    apps, worker_amounts = parse('small_strong_scaling_data/app-*', workers)
    return tuple(apps), tuple(worker_amounts)

### Some convenient functions ###
    
def plt_setup(xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ymin=0)
    
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

def calc_r2_fn(xs, ys, fn):
    model_ys = (fn(x) for x in xs)
    return calc_r2(xs, ys, model_ys)
# pylint: disable=invalid-name, missing-docstring
import glob
from sparklogstats import LogParser
from itertools import chain
import numpy as np


class Experiment:
    def __init__(self, files, n_nodes=None):
        # apps indexed by number of nodes
        self.apps = {}
        self.n_nodes = None
        self.threads = 2
        self.n_apps = None
        self._parse(files, n_nodes)

    def get_apps(self, n_nodes=None):
        if n_nodes is None:
            n_nodes = self.n_nodes
        apps = (self.apps[nn] for nn in n_nodes)
        return chain.from_iterable(apps)

    def reject_stage_outliers(self, stage, std_m=6):
        for nn, apps in self.apps.items():
            durs = np.array([a.stages[stage].duration for a in apps])
            mean, std = durs.mean(), durs.std()
            remove_outliers(apps, stage, mean, std_m * std, nn)

    def get_stage_n_nodes(self):
        return (a.slaves for a in self.get_apps())

    def get_stage_durations(self, stage):
        return (a.stages[stage].duration for a in self.get_apps())

    def get_stage_duration_lists(self, stage):
        return ((a.stages[stage].duration for a in apps_by_nn)
                for apps_by_nn in self._get_app_lists())

    def get_task_n_nodes(self, stage, n_nodes=None):
        if n_nodes is None:
            n_nodes = self.n_nodes

        def repeat_executions_iter():
            for app in self.get_apps():
                for _ in app.stages[stage].tasks:
                    if app.slaves in n_nodes:
                        yield app.slaves

        return repeat_executions_iter()

    def get_task_durations(self, stage, n_nodes=None):
        if n_nodes is None:
            n_nodes = self.n_nodes

        return (t.duration for a in self.get_apps()
                for t in a.stages[stage].tasks if a.slaves in n_nodes)

    # The lists have to be in order to be plotted as lines

    def get_task_duration_lists(self, stage):
        def get_all_tasks(app):
            return app.stages[stage].tasks

        return self._get_selected_tasks_duration_lists(get_all_tasks)

    def get_first_task_duration_lists(self, stage, n_nodes=None):
        def get_first_tasks(app):
            workers = app.slaves * self.threads
            tasks = app.stages[stage].tasks
            return tasks[:workers]

        return self._get_selected_tasks_duration_lists(get_first_tasks,
                                                       n_nodes)

    def get_first_task_durations(self, stage, n_nodes=None):
        dur_lists = self.get_first_task_duration_lists(stage, n_nodes)
        return chain.from_iterable(dur_lists)

    def get_nonfirst_task_duration_lists(self, stage, n_nodes=None):
        def get_nonfirst_tasks(app):
            workers = app.slaves * self.threads
            tasks = app.stages[stage].tasks
            return tasks[workers:]

        return self._get_selected_tasks_duration_lists(get_nonfirst_tasks,
                                                       n_nodes)

    def get_nonfirst_task_durations(self, stage, n_nodes=None):
        dur_lists = self.get_nonfirst_task_duration_lists(stage, n_nodes)
        return chain.from_iterable(dur_lists)

    def _get_selected_tasks_duration_lists(self, select, n_nodes=None):
        return (
            chain.from_iterable(
                (t.duration for t in select(app))
                for app in apps_by_nn
            )
            for apps_by_nn in self._get_app_lists(n_nodes))

    def get_first_task_n_nodes(self, stage, n_nodes=None):
        dur_lists = self.get_first_task_duration_lists(stage, n_nodes)
        return self._get_n_nodes_from_dur_lists(dur_lists, n_nodes)

    def get_nonfirst_task_n_nodes(self, stage, n_nodes=None):
        dur_lists = self.get_nonfirst_task_duration_lists(stage, n_nodes)
        return self._get_n_nodes_from_dur_lists(dur_lists, n_nodes)

    def _get_n_nodes_from_dur_lists(self, dur_lists, n_nodes=None):
        if n_nodes is None:
            n_nodes = self.n_nodes

        def iterator():
            for nn, durs in zip(n_nodes, dur_lists):
                for _ in durs:
                    yield nn

        return iterator()

    def _get_app_lists(self, n_nodes=None):
        if n_nodes is None:
            n_nodes = self.n_nodes
        return (self.apps[nn] for nn in n_nodes)

    def _parse(self, files, n_nodes=None):
        parser = LogParser()
        for log in glob.glob(files):
            parser.parse_file(log)
            app = parser.app
            self._add_app(app, n_nodes)
        self._finish_parsing()

    def _add_app(self, app, n_nodes=None):
        # Ignoring list of hostnames. Only the amount matters.
        app.slaves = len(app.slaves)
        if not n_nodes or app.slaves in n_nodes:
            self.apps.setdefault(app.slaves, []).append(app)

    def _finish_parsing(self):
        self.n_nodes = tuple(sorted(self.apps.keys()))
        self.n_apps = sum(len(apps) for apps in self.apps.values())


def remove_outliers(apps, stage, mean, threshold, n_nodes):
    for app in apps:
        dur = app.stages[stage].duration
        if abs(dur - mean) > threshold:
            apps.remove(app)
            print('INFO: Removed outlier ({:d}, {:g}).'.format(n_nodes, dur))

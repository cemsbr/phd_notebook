"""Information about number of tasks and input size."""
from lib.bundler import BaseBundle
from lib.csv_gen import CSVGen
from lib.hbkmeans_parser import HBKmeansParser
from lib.hbsort_parser import HBSortParser
from lib.wikipedia_parser import WikipediaParser


class Bundle(BaseBundle):
    """Information about number of tasks and input size."""

    def __init__(self):
        """Filename to be generated."""
        super().__init__('input_tasks.csv')
        self._csv_gen = CSVGen()
        self._writer = None

    def run(self):
        """Parse all applications."""
        self.start()

        header = ('Application', 'Size (MB)', '1st-Stage Tasks')
        self._writer = self._csv_gen.get_writer(header, self.filename)

        all_apps = (
            ('hbsort', HBSortParser.get_apps()),
            ('hbkmeans', HBKmeansParser.get_apps()),
            ('wikipedia', WikipediaParser().get_apps())
        )
        for name, apps in all_apps:
            self._write(name, apps)

        self._csv_gen.close()
        self.finish()

    def _write(self, name, apps):
        MB = 1024**2
        for el in apps:
            #: Wikipedia get_apps return a tuple with corrected size.
            app, size = el if name == 'wikipedia' else (el, el.bytes_read)
            # Counting elements of the generator
            tasks = sum(1 for _ in app.stages[0].successful_tasks)
            self._writer.writerow((name, size / MB, tasks))


if __name__ == '__main__':
    Bundle().update()

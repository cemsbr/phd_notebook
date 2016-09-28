"""Ease the creation of CSV files."""
import csv


class CSVGen:
    """Help creating CSV files."""

    def __init__(self):
        """Keep track of open files to be closed later."""
        self._open_files = []

    def get_writers(self, header, filenames):
        """Return CSV writers with header already written."""
        return [self.get_writer(header, f) for f in filenames]

    def get_writer(self, header, filename):
        """Return CSV writer with header already written.

        If header is None, nothing is written.
        """
        file = open(filename, 'w', newline='')
        self._open_files.append(file)
        writer = csv.writer(file)
        if header is not None:
            writer.writerow(header)
        return writer

    def close(self):
        """Close open files. Do no harm if there is none."""
        while self._open_files:
            self._open_files.pop().close()

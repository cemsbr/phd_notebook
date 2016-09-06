"""Module to manage bundles."""
import sys
from importlib import import_module
from inspect import currentframe, getouterframes
from os import path


__all__ = ('Bundler')


class Bundler:
    """Manage bundles, e.g. get files from other bundles."""

    bundles_root = None

    @classmethod
    def set_bundles_root(cls, *walk2bundles):
        """Need the path where the bundles are.

        Args:
            walk2bundles: Folders to append to current dir and arrive at
                your bundles' root folder.

        Example:
            .. code-block:: python

                Bundler.set_bundles_root('..', 'bundles')  # for `../bundles`.
        """
        caller = getouterframes(currentframe())[1]
        caller_dir = path.dirname(caller.filename)
        cls.bundles_root = path.join(caller_dir, *walk2bundles)

    @classmethod
    def get_file(cls, dir_name, filename):
        """Get the path for a bundle's file. Run the bundle if necessary.

        Args:
            dir_name (str): bundle dir name.
            filename (str): basename of the bundle's file.

        Returns:
            str: path to filename.
        """
        bundle = cls.get_bundle(dir_name)
        return bundle.get_file(filename)

    @classmethod
    def get_bundle(cls, dir_name):
        """Return a bundle by its dir name.

        Returns:
            Bundle: bundle inside `dir_name` folder.
        """
        try:
            sys.path.insert(0, cls.bundles_root)
            bundle_mod = import_module(dir_name + '.bundle')
            bundle_cls = getattr(bundle_mod, 'Bundle')
            return bundle_cls()
        finally:  # make sure the path is removed
            sys.path.pop(0)

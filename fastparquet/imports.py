import importlib


class NotImported:
    dummy = True

    def __init__(self, err):
        self.err = err
        self.__version__ = "0"

    def __getattr__(self, item):
        raise AttributeError(item) from self.err


def __getattr__(name):
    try:
        mod = importlib.import_module(name)
    except ImportError as ex:
        mod = NotImported(ex)
    globals()[name] = mod

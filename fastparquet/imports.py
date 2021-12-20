

class NotImported:
    dummy = True

    def __init__(self, err):
        self.err = err
        self.__version__ = "0"

    def __getattr__(self, item):
        raise AttributeError(item) from self.err


try:
    import pandas as pd
except ImportError as ex:
    pd = NotImported(ex)


try:
    import awkward as ak
except ImportError as ex:
    ak = NotImported(ex)

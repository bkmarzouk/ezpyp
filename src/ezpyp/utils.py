import os
from typing import Callable


class MissingDependency(Exception):
    pass


class RepeatedStepError(Exception):
    pass


class SchemaConflictError(Exception):
    pass


class InitializationError(Exception):
    pass


class NotExecutedStepWarning(UserWarning):
    pass


class UnrecognizedStepWarning(UserWarning):
    pass


class SkippedStepWarning(UserWarning):
    pass


class FailedStepWarning(UserWarning):
    pass


_EZPYP_HASH = str(3743294)


def track_function_source():
    return os.environ.get("_TRACK_FUNCTION_SOURCE", "False") == "True"


def fixed_hash(function):
    def tmp_fixed_hash(*args, **kwargs):
        os.environ["PYTHONHASHSEED"] = _EZPYP_HASH
        out = function(*args, **kwargs)
        del os.environ["PYTHONHASHSEED"]
        return out

    return tmp_fixed_hash


@fixed_hash
def hash_function(function: Callable):
    return hash(function)


class _Comm:
    @staticmethod
    def Barrier():
        pass

    @staticmethod
    def Get_rank():
        return 0

    @staticmethod
    def Get_size():
        return 1


class _MPI:
    COMM_WORLD = _Comm()

    @staticmethod
    def Finalize():
        pass

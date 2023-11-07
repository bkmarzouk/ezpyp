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


def item_selection(
    items_to_distribute: list | dict, mpi_rank: int, mpi_size: int
):
    if isinstance(items_to_distribute, dict):
        items_to_distribute = list(items_to_distribute.keys())

    if mpi_size == 1:
        return items_to_distribute

    n_items_to_distribute = len(items_to_distribute)
    min_items_per_proc = (
        n_items_to_distribute // mpi_size
    )  # Populates list with min. number of items chose by int. div.
    rem_items_to_distribute = (
        n_items_to_distribute % mpi_size
    )  # Find num. of remaining items with mod. div. and pop. lists

    items_for_current_proc = []

    for i in range(mpi_size):
        items_for_current_proc += items_to_distribute[
            i * min_items_per_proc : (i + 1) * min_items_per_proc
        ]

    if rem_items_to_distribute > 0:
        for j in range(rem_items_to_distribute):
            extra = items_to_distribute[min_items_per_proc * mpi_size + j]
            items_for_current_proc[j].append(extra)

    return items_for_current_proc[mpi_rank]


def as_single_process(
    function: Callable, current_proc: int, executing_proc: int
):
    def wrapper(*args, **kwargs):
        if current_proc == executing_proc:
            return function(*args, **kwargs)

    return wrapper

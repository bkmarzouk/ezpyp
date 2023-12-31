import os
from copy import deepcopy
from typing import Callable
from ezpyp.mpi import MPI


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


def item_selection(
    items_to_distribute: list | dict, mpi_rank: int, mpi_size: int
) -> list:
    if isinstance(items_to_distribute, dict):
        items_to_distribute = list(items_to_distribute.keys())

    if mpi_size == 1:
        return items_to_distribute

    items_to_distribute = deepcopy(items_to_distribute)

    current_idx = 0
    output = {k: [] for k in range(mpi_size)}

    while items_to_distribute:
        if current_idx == mpi_size:
            current_idx = 0

        output[current_idx].append(items_to_distribute.pop(0))

        current_idx += 1

    return output[mpi_rank]


def as_single_process(
    current_proc: int, mpi_comm: MPI.COMM_WORLD, executing_proc: int = 0
):
    def decorator(function: Callable):
        def wrapper(*args, **kwargs):
            if current_proc == executing_proc:
                result = function(*args, **kwargs)
            else:
                result = None
            mpi_comm.Barrier()
            return result

        return wrapper

    return decorator


def get_skip_downstream_condition(steps: list):
    return any([s.get_status() > 0 for s in steps])

import inspect
from copy import deepcopy
from pathlib import Path
from typing import List, Any, Callable

from ezpyp.cache import PickleCache, DillCache, NumpyCache
from ezpyp.utils import fixed_hash, MissingDependency, track_function_source


class _Step:
    def __init__(
        self,
        cache_location: Path,
        name: str,
        args: List[Any],
        kwargs: dict,
        function: Callable,
        depends_on: List[Any],
        extra_suffix: str,
    ):
        self.cache_location = cache_location
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.depends_on = depends_on
        self.step_ext = (
            "step" if extra_suffix == "" else f"step.{extra_suffix}"
        )
        self.status_ext = (
            "status" if extra_suffix == "" else f"status.{extra_suffix}"
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    @fixed_hash
    def __hash__(self):
        schema_str = str(self.get_schema())
        return hash(schema_str)

    @staticmethod
    def _load_object(object_path: Path) -> Any:
        raise FileNotFoundError("No file can exist for this base class.")

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
        raise FileNotFoundError("No file can exist for this base class.")

    def _get_object_path(self, extension: str):
        object_path = self.cache_location.joinpath(self.name).with_suffix(
            f".{extension}"
        )
        return object_path

    def _results_path(self):
        return self._get_object_path(extension=self.step_ext)

    def _status_path(self):
        return self._get_object_path(extension=self.status_ext)

    def _load_result(self):
        return self._load_object(self._results_path())

    def _load_status(self):
        return self._load_object(self._status_path())

    def _cache_result(self, result: Any):
        self._cache_object(self._results_path(), result)

    def _cache_status(self, status: int):
        self._cache_object(self._status_path(), status)

    def _mark_as_skipped(self):
        self._cache_status(2)

    def check_ready(self):
        for step in self.depends_on:
            dep_status = step.get_status()
            if dep_status not in (0, 2):
                raise MissingDependency(
                    f"Dependent step '{step}' has non-zero status: {dep_status}"
                )

    def _update_step_arg_values(self):
        for index, arg in enumerate(self.args):
            if isinstance(arg, PlaceHolder):
                self.args[index] = arg.get_result()

        for key, value in self.kwargs.items():
            if isinstance(value, PlaceHolder):
                self.kwargs[key] = value.get_result()

    def execute(self, _skip=False):
        if _skip:
            result = None
            status = 2
            print(f"  ╰─> [SKIP] '{self.name}' - dependency failure detected.")
        else:
            self.check_ready()
            self._update_step_arg_values()
            try:
                result = self.function(*self.args, **self.kwargs)
                status = 0
                print(f"  ╰─> [PASS] '{self.name}'")
                self._cache_result(result)
            except Exception as exception:
                result = None
                status = 1
                print(f"  ╰─> [FAIL] '{self.name}'\n       {exception}")

        self._cache_status(status)
        return result

    def get_result(self):
        try:
            result = self._load_result()
            # print(f"Result from step '{self.name}' loaded successfully")
            return result
        except FileNotFoundError:
            return self.execute()

    def get_status(self):
        try:
            status = self._load_status()
            # print(f"Status from step '{self.name}' loaded successfully")
            return status
        except FileNotFoundError:
            return -1

    def get_schema(self):
        core = deepcopy(self.__dict__)

        # More useful function defs
        core["cache_location"] = str(core["cache_location"])
        core["function_name"] = core["function"].__name__
        core["function_bytes"] = core["function"].__code__.co_code.__str__()

        if track_function_source():
            core["function_source"] = inspect.getsource(self.function)
        # core["function_hash"] = hash_function(self.function) #... nope
        del core["function"]

        # Make sense of step and placeholder instances
        for key in ("args", "kwargs", "depends_on"):
            core[key] = repr(core[key])

        return core

    def simple_summary(self, phase):
        keys = ("name", "args", "kwargs")
        step_schema = self.get_schema()
        data_to_summarize = {k: step_schema[k] for k in keys}
        data_to_summarize["phase"] = str(phase)
        data_to_summarize["status"] = str(self.get_status())
        data_to_summarize["result"] = (
            str(self._results_path())
            if self._results_path().exists()
            else "N/A"
        )

        return "\t\t".join(data_to_summarize.values()) + "\n"


class PickleStep(PickleCache, _Step):
    def __init__(
        self,
        cache_location: Path,
        name: str,
        args: List[Any],
        kwargs: dict,
        function: Callable,
        depends_on: List[Any],
    ):
        super().__init__(
            cache_location=cache_location,
            name=name,
            args=args,
            kwargs=kwargs,
            function=function,
            depends_on=depends_on,
            extra_suffix="",
        )

    def __str__(self):
        return f"Pickle Step Object '{self.name}'"

    def __repr__(self):
        return str(self)


class DillStep(DillCache, _Step):
    def __init__(
        self,
        cache_location: Path,
        name: str,
        args: List[Any],
        kwargs: dict,
        function: Callable,
        depends_on: List[Any],
    ):
        super().__init__(
            cache_location=cache_location,
            name=name,
            args=args,
            kwargs=kwargs,
            function=function,
            depends_on=depends_on,
            extra_suffix="",
        )

    def __str__(self):
        return f"Dill Step Object '{self.name}'"

    def __repr__(self):
        return str(self)


class NumpyStep(NumpyCache, _Step):
    def __init__(
        self,
        cache_location: Path,
        name: str,
        args: List[Any],
        kwargs: dict,
        function: Callable,
        depends_on: List[Any],
    ):
        super().__init__(
            cache_location=cache_location,
            name=name,
            args=args,
            kwargs=kwargs,
            function=function,
            depends_on=depends_on,
            extra_suffix="npy",
        )

    def __str__(self):
        return f"Numpy Step Object '{self.name}'"

    def __repr__(self):
        return str(self)


class PlaceHolder:
    def __init__(self, step: _Step):
        self.name = step.name
        self._update = step.get_result

    def __str__(self):
        return f"TMP for {self.name}"

    def __repr__(self):
        return f"TMP for {self.name}"

    def get_result(self):
        return self._update()

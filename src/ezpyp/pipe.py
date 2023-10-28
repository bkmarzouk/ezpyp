from typing import Any, List, Callable
from pathlib import Path
import pickle
import dill
import numpy as np


class MissingDependency(Exception):
    pass


class _Cache:
    @staticmethod
    def _load_object(object_path: Path):
        pass

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
        pass


class PickleCache(_Cache):
    @staticmethod
    def _load_object(object_path: Path):
        print("pickle")
        with open(object_path, "rb") as f:
            cached_object = pickle.load(f)

        return cached_object

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
        print("pickle")
        with open(object_path, "wb") as f:
            pickle.dump(object_to_cache, f)


class DillCache(_Cache):
    @staticmethod
    def _load_object(object_path: Path):
        with open(object_path, "rb") as f:
            cached_object = dill.load(f)

        return cached_object

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
        with open(object_path, "wb") as f:
            dill.dump(object_to_cache, f)


class NumpyCache(_Cache):
    @staticmethod
    def _load_object(object_path: Path):
        np.load(object_path)

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
        np.save(object_path, object_to_cache)


class _Step:
    def __init__(
        self,
        cache_location: Path,
        name: str,
        args: List[Any],
        kwargs: dict,
        function: Callable,
        depends_on: List[Any],
    ):
        self.cache_location = cache_location
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.depends_on = depends_on
        try:
            self.status = self.load_status()
        except FileNotFoundError:
            self.status = -1

    @staticmethod
    def _load_object(object_path: Path) -> Any:
        pass

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
        pass

    def _get_object_path(self, extension: str):
        object_path = self.cache_location.joinpath(self.name).with_suffix(
            f".{extension}"
        )
        return object_path

    def get_results_path(self):
        return self._get_object_path(extension="step")

    def get_status_path(self):
        return self._get_object_path(extension="status")

    def load_result(self):
        return self._load_object(self.get_results_path())

    def load_status(self):
        return self._load_object(self.get_status_path())

    def cache_result(self, result: Any):
        self._cache_object(self.get_results_path(), result)

    def cache_status(self, status: int):
        self._cache_object(self.get_status_path(), status)

    def check_ready(self):
        for step in self.depends_on:
            if not step.status == 0:
                raise MissingDependency(step)

    def execute(self):
        try:
            result = self.function(*self.args, **self.kwargs)
            status = 0
            print(f"[PASS] '{self.name}'")
        except Exception as e:
            result = None
            status = 1
            print(f"[FAIL] '{self.name}'\n       {e}")

        self.cache_result(result)
        self.cache_status(status)
        self.status = status
        return result

    def get_result(self):
        try:
            result = self.load_result()
            print(f"Result from step '{self.name}' loaded successfully")
            return result
        except FileExistsError:
            return self.execute()


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
        )


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
        )


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
        )


class Pipeline:
    def __init__(self, cache_location: Path, pipeline_id: str):
        self.cache_location = cache_location
        self.pipeline_id = pipeline_id

    def add_pickle_step(self, *args, **kwargs):
        pass  # TODO

    def add_dill_step(self, *args, **kwargs):
        pass  # TODO

    def add_numpy_step(self, *args, **kwargs):
        pass  # TODO

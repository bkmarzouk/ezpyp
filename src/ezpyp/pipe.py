from copy import deepcopy
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
        with open(object_path, "rb") as f:
            cached_object = pickle.load(f)

        return cached_object

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
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
        return np.load(object_path)

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

    def __eq__(self, other):
        # TODO: This should be improved at some point but will help with
        #       uniqueness at least for now.
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

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


class PlaceHolder:
    def __init__(self, step: _Step):
        self.name = step.name
        self._update = step.get_result

    def __str__(self):
        return f"TMP for {self.name}"

    def __repr__(self):
        return f"TMP for {self.name}"

    def update(self):
        return self._update()


class RepeatedStepError(Exception):
    pass


class Pipeline:
    def __init__(self, cache_location: Path, pipeline_id: str):
        self.phases = None
        self.cache_location = cache_location
        self.pipeline_id = pipeline_id
        self.steps = []

    def add_step(self, step: PickleStep | DillStep | NumpyStep):
        if step in self.steps:
            raise RepeatedStepError(
                f"Step '{step.name}' already detected in pipeline!"
            )

        self.steps.append(step)

    def organize_steps(self):
        phases = {}

        for step in self.steps:
            n = len(set(step.depends_on))

            if n in phases:
                phases[n].append(step)
            else:
                phases[n] = [step]

        # Remap phases such that indices are strictly monotonic and linear
        for ii in range(len(phases)):
            if ii not in phases:
                jj = ii + 1

                while jj not in phases:
                    jj += 1

                phases[ii] = phases[jj]
                del phases[jj]

        self.phases = phases


def expand_dependencies(dependencies: List[PickleStep | DillStep | NumpyStep]):
    expanded = []

    while dependencies:
        for dep_step in dependencies:
            expanded.append(dep_step)
            dependencies.remove(dep_step)
            if dep_step.depends_on:
                expanded += expand_dependencies(dep_step.depends_on)

    return expanded


def _as_step(
    step_type: str,
    pipeline: Pipeline,
    depends_on: List[PickleStep | DillStep | NumpyStep] = [],
):
    def decorator(function):
        def wrapper(*args, **kwargs):
            step_class = {
                "pickle": PickleStep,
                "dill": DillStep,
                "numpy": NumpyStep,
            }[step_type]

            pipeline_step: PickleStep | DillStep | NumpyStep = step_class(
                cache_location=pipeline.cache_location,
                name=function.__name__,
                args=list(args),
                kwargs=kwargs,
                function=function,
                depends_on=expand_dependencies(deepcopy(depends_on)),
            )

            pipeline.add_step(pipeline_step)

            return pipeline_step

        return wrapper

    return decorator


def as_pickle_step(
    pipeline: Pipeline,
    depends_on: List[PickleStep | DillStep | NumpyStep] = [],
):
    return _as_step("pickle", pipeline, depends_on)


def as_dill_step(
    pipeline: Pipeline,
    depends_on: List[PickleStep | DillStep | NumpyStep] = [],
):
    return _as_step("dill", pipeline, depends_on)


def as_numpy_step(
    pipeline: Pipeline,
    depends_on: List[PickleStep | DillStep | NumpyStep] = [],
):
    return _as_step("numpy", pipeline, depends_on)


if __name__ == "__main__":
    pipeline = Pipeline(Path.cwd(), "test")

    @as_pickle_step(pipeline)
    def a():
        return 1

    step_a = a()

    @as_pickle_step(pipeline)
    def b():
        return 2

    step_b = b()

    @as_pickle_step(pipeline, depends_on=[step_a, step_b])
    def c(alpha, beta):
        return alpha + beta

    step_c = c(PlaceHolder(step_a), PlaceHolder(step_b))

    @as_pickle_step(pipeline, depends_on=[step_c])
    def d(gamma, n):
        return gamma**n

    step_d = d(PlaceHolder(step_c), 3)

    @as_pickle_step(pipeline, depends_on=[step_d, step_a])
    def e(delta, alpha):
        return delta * alpha

    step_e = e(PlaceHolder(step_d), PlaceHolder(step_a))

    @as_pickle_step(pipeline, depends_on=[step_c])
    def x(gamma, q):
        return gamma - q

    step_x = x(PlaceHolder(step_c), 34)

    @as_pickle_step(pipeline, depends_on=[step_x])
    def y(chi):
        return chi / 2

    step_y = y(PlaceHolder(step_x))

    @as_pickle_step(pipeline, depends_on=[step_e, step_x])
    def z(eta, chi):
        return eta / chi

    step_z = z(PlaceHolder(step_e), PlaceHolder(step_x))

    pipeline.organize_steps()

    print(pipeline.phases)

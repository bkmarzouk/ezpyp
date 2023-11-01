import inspect
from copy import deepcopy
from typing import Any, List, Callable, Dict
from pathlib import Path
import pickle
import dill
import numpy as np
import json
from warnings import warn

from ezpyp.utils import (
    MissingDependency,
    RepeatedStepError,
    SchemaConflictError,
    InitializationError,
    NotExecutedStepWarning,
    UnrecognizedStepWarning,
    SkippedStepWarning,
    FailedStepWarning,
    track_function_source,
    fixed_hash,
)


class _Cache:
    _SERIALIZATION: str

    @staticmethod
    def _load_object(object_path: Path):
        pass

    @staticmethod
    def _cache_object(object_path: Path, object_to_cache: Any):
        pass


class PickleCache(_Cache):
    _SERIALIZATION = "pickle"

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
    _SERIALIZATION = "dill"

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
    _SERIALIZATION = "numpy"

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


class _Pipeline:
    def __init__(self, cache_location: Path, pipeline_id: str):
        self.phases: Dict[int, List[PickleStep | DillStep | NumpyStep]] = {}
        self.cache_location = cache_location
        self.pipeline_id = pipeline_id
        self.steps: List[PickleStep | DillStep | NumpyStep] = []
        self.schema_path = cache_location.joinpath("pipeline.schema")
        self.summary_path = cache_location.joinpath("pipeline.summary")
        self._initialized_schema = False
        self._execution_complete = False
        self._pipeline_failed = False

    def add_step(self, step: PickleStep | DillStep | NumpyStep):
        if self._initialized_schema:
            raise InitializationError(
                "Pipeline already initialized with "
                f"schema {self.schema_path} - no more "
                f"steps can be added."
            )

        if step in self.steps:
            raise RepeatedStepError(
                f"Step '{step.name}' already detected in pipeline!"
            )

        # print(f"Adding step... {step}")
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

    def execute(self, *args, **kwargs):
        pass

    def lock_step(self, *args, **kwargs):
        pass

    def unlock_step(self, *args, **kwargs):
        pass

    def write_error_report(self, *args, **kwargs):
        pass

    def get_schema(self):
        phases_schema: Dict[str, List[Dict[str, str]]] = {}
        for key, steps in self.phases.items():
            phases_schema[str(key)] = [s.get_schema() for s in steps]

        # pipeline_schema = {"pipeline_id": self.pipeline_id}.update(phases_schema)

        if phases_schema == {}:
            raise InitializationError("Schema contains no data from phases")

        return phases_schema  # pipeline_schema

    def initialize_schema(self):
        self.organize_steps()

        current_schema = self.get_schema()

        if self.schema_path.exists():
            with open(self.schema_path, "r") as f:
                existing_schema = json.load(f)

            if current_schema.keys() != existing_schema.keys():
                raise SchemaConflictError(
                    f"Existing schema contains a different number of "
                    f"computational phases compared to those defined in this "
                    f"pipeline: {current_schema.keys()} != "
                    f"{existing_schema.keys}"
                )

            for phase_index in current_schema:
                for step_schema in current_schema[phase_index]:
                    if step_schema not in existing_schema[phase_index]:
                        raise SchemaConflictError(
                            f"Schema step '{step_schema}' does not belong to "
                            f"existing schema file '{self.schema_path}'.\n"
                            f"In order to run the pipeline with different "
                            f"definitions define a different cache_location, "
                            f"or clear the existing cache data."
                        )

        with open(self.schema_path, "w") as f:
            json.dump(current_schema, f, indent=2, sort_keys=True)

        self._initialized_schema = True

    def finalize(self, *args, **kwargs):
        pass

    @fixed_hash
    def __hash__(self):
        schema_str = str(self.get_schema())
        return hash(schema_str)


class SerialPipeline(_Pipeline):
    def __init__(self, cache_location: Path, pipeline_id: str):
        super().__init__(cache_location, pipeline_id)

    def get_lock_data(
        self,
        current_phase: int,
        active_step: PickleStep | DillStep | NumpyStep,
    ):
        lock_data = {
            "scheme_hash": hash(self),
            "phase": current_phase,
            "active": hash(active_step),
        }
        return lock_data

    def get_lock_path(self, done=False):
        return self.cache_location.joinpath("step").with_suffix(
            ".done" if done else ".active"
        )

    def lock_step(
        self,
        current_phase: int,
        active_step: PickleStep | DillStep | NumpyStep,
    ):
        with open(self.get_lock_path(), "w") as f:
            lock_data = self.get_lock_data(current_phase, active_step)
            json.dump(lock_data, f, indent=2)

    def unlock_step(self):
        self.get_lock_path().rename(self.get_lock_path(done=True))

    def execute(self, *args, **kwargs):
        self.initialize_schema()
        self.organize_steps()

        skip_downstream_phases = False

        for phase_index in sorted(self.phases.keys()):
            print(
                "[Executing pipeline phase %02d/%02d]"
                % (phase_index, len(self.phases) - 1)
            )

            current_phase = self.phases[phase_index]
            error_in_current_phase = False

            for step in current_phase:
                self.lock_step(phase_index, step)

                step.execute(
                    _skip=skip_downstream_phases - error_in_current_phase
                )

                if step.get_status() == 1:
                    skip_downstream_phases = True
                    error_in_current_phase = True
                    self._pipeline_failed = True

                self.unlock_step()

        self._execution_complete = True
        self.finalize()

    def finalize(self, *args, **kwargs):
        assert self._execution_complete

        desc = "Failed" if self._pipeline_failed else "Completed"

        header = f"Pipeline {desc} @ {self.cache_location}\n\n"

        cols = (
            "\t\t".join(
                ["name", "args", "kwargs", "phase", "status", "result"]
            )
            + "\n"
        )

        with open(self.summary_path, "w") as f:
            f.write(header)
            f.write(cols)

            for phase, phase_steps in self.phases.items():
                for step in phase_steps:
                    f.write(step.simple_summary(phase))

    def get_result(self, step_name: str):
        names = []

        for step in self.steps:
            if step.name == step_name:
                try:
                    # print(f"attempting to load result for {step}")
                    return step._load_result()
                except FileNotFoundError:
                    # print(f"FileNotFound for {step}")
                    step_status = step.get_status()

                    if step_status == -1:
                        warn(
                            "Pipeline step not yet performed: data cannot be "
                            "retrieved... returning None",
                            NotExecutedStepWarning,
                        )
                    elif step_status == 1:
                        warn(
                            "Pipeline step failed: data cannot be retrieved"
                            "... returning None",
                            FailedStepWarning,
                        )
                    elif step_status == 2:
                        warn(
                            "Pipeline step skipped: dependency failure"
                            "... returning None",
                            SkippedStepWarning,
                        )
                    else:
                        raise FileNotFoundError(step)
                    return None

            names.append(step.name)

        warn(
            "Step name not found in pipeline... returning None",
            UnrecognizedStepWarning,
        )
        return None


def expand_dependencies(dependencies: List[PickleStep | DillStep | NumpyStep]):
    # Find all nested dependencies (duplicates may exist)

    dependencies = deepcopy(dependencies)

    expanded = []

    while dependencies:
        for dep_step in dependencies:
            expanded.append(dep_step)
            dependencies.remove(dep_step)
            if dep_step.depends_on:
                expanded += expand_dependencies(dep_step.depends_on)

    return expanded


def reduce_dependencies(dependencies: List[PickleStep | DillStep | NumpyStep]):
    # Remove duplicates from dependency expansion

    indices_to_drop = []
    n_deps = len(dependencies)
    for ii in range(n_deps - 1):
        dep = dependencies[ii]
        if dep in dependencies[ii + 1 :]:
            indices_to_drop.append(ii)

    for ii in indices_to_drop[::-1]:
        # print(f"Removing duplicated dependency {dependencies.pop(ii)}")
        dependencies.pop(ii)

    return dependencies


def simplify_dependencies(
    dependencies: List[PickleStep | DillStep | NumpyStep],
):
    return reduce_dependencies(expand_dependencies(dependencies))


def _as_step(
    step_type: str,
    pipeline: _Pipeline,
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
                depends_on=simplify_dependencies(depends_on),
            )

            pipeline.add_step(pipeline_step)

            return pipeline_step

        return wrapper

    return decorator


def as_pickle_step(
    pipeline: _Pipeline,
    depends_on: List[PickleStep | DillStep | NumpyStep] = [],
):
    return _as_step("pickle", pipeline, depends_on)


def as_dill_step(
    pipeline: _Pipeline,
    depends_on: List[PickleStep | DillStep | NumpyStep] = [],
):
    return _as_step("dill", pipeline, depends_on)


def as_numpy_step(
    pipeline: _Pipeline,
    depends_on: List[PickleStep | DillStep | NumpyStep] = [],
):
    return _as_step("numpy", pipeline, depends_on)

import json
from pathlib import Path
from typing import Dict, List

from _warnings import warn

from ezpyp.steps import DillStep, NumpyStep, PickleStep, simplify_dependencies
from ezpyp.utils import (
    FailedStepWarning,
    InitializationError,
    NotExecutedStepWarning,
    RepeatedStepError,
    SchemaConflictError,
    SkippedStepWarning,
    UnrecognizedStepWarning,
    fixed_hash,
    item_selection,
    get_skip_downstream_condition,
    as_single_process,
)
from ezpyp.mpi import MPI

COMM = MPI.COMM_WORLD
MPI_RANK = COMM.Get_rank()
MPI_SIZE = COMM.Get_size()


class Pipeline:
    def __init__(self, cache_location: Path | str, pipeline_id: str):
        if isinstance(cache_location, str):
            cache_location = Path(cache_location)

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

    @staticmethod
    def barrier():
        COMM.Barrier()

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

    def write_error_report(self, *args, **kwargs):
        pass

    def get_schema(self):
        phases_schema: Dict[str, List[Dict[str, str]]] = {}
        for key, phase_steps in self.phases.items():
            phases_schema[str(key)] = [s.get_schema() for s in phase_steps]

        # pipeline_schema = {"pipeline_id": self.pipeline_id}.update(phases_schema)

        if phases_schema == {}:
            raise InitializationError("Schema contains no data from phases")

        return phases_schema  # pipeline_schema

    def initialize_schema(self):
        self.organize_steps()

        if MPI_RANK == 0:
            current_schema = self.get_schema()
            error = None
            if self.schema_path.exists():
                with open(self.schema_path, "r") as f:
                    existing_schema = json.load(f)

                if current_schema.keys() != existing_schema.keys():
                    error = SchemaConflictError(
                        f"Existing schema contains a different number of "
                        f"computational phases compared to those defined in this "
                        f"pipeline: {current_schema.keys()} != "
                        f"{existing_schema.keys}"
                    )
                else:
                    for phase_index in current_schema:
                        for step_schema in current_schema[phase_index]:
                            if step_schema not in existing_schema[phase_index]:
                                error = SchemaConflictError(
                                    f"Schema step '{step_schema}' does not belong to "
                                    f"existing schema file '{self.schema_path}'.\n"
                                    f"In order to run the pipeline with different "
                                    f"definitions define a different cache_location, "
                                    f"or clear the existing cache data."
                                )

            if error is None:
                with open(self.schema_path, "w") as f:
                    json.dump(current_schema, f, indent=2, sort_keys=True)

            for ii in range(1, MPI_SIZE):
                COMM.send(error, dest=ii)

        else:
            error = COMM.recv(source=0)

        if error is not None:
            raise error

        COMM.Barrier()

        self._initialized_schema = True

    @fixed_hash
    def __hash__(self):
        schema_str = str(self.get_schema())
        return hash(schema_str)

    def get_lock_path(self, done=False):
        process_id = "_proc_%03d" % MPI_RANK
        return self.cache_location.joinpath(f"step{process_id}").with_suffix(
            ".done" if done else ".active"
        )

    def execute(self):
        self.initialize_schema()
        self.organize_steps()

        skip_downstream_phases = False

        for phase_index in sorted(self.phases.keys()):
            current_phase_steps = self.phases[phase_index]
            error_in_current_phase = False

            steps_for_process = item_selection(
                current_phase_steps, MPI_RANK, MPI_SIZE
            )

            for step in steps_for_process:
                print(
                    f"[Executing step '{step}' of phase {phase_index}] [MPI process @ {MPI_RANK}]"
                )

                # print(self.get_lock_path())

                self.lock_step(phase_index, step)

                step.execute(
                    _skip=skip_downstream_phases - error_in_current_phase
                )

                if step.get_status() == 1:
                    skip_downstream_phases = True
                    error_in_current_phase = True
                    self._pipeline_failed = True

                self.unlock_step()

            COMM.Barrier()
            skip_downstream_phases = get_skip_downstream_condition(
                current_phase_steps
            )

        self._execution_complete = True
        self.finalize()

    @as_single_process(MPI_RANK, mpi_comm=COMM)
    def finalize(self):
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
                    return step._load_result()
                except FileNotFoundError:
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
                depends_on=simplify_dependencies(depends_on),
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
    print(MPI_RANK + 1, "/", MPI_SIZE)

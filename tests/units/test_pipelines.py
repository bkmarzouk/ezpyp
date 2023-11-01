import pytest

from ezpyp.pipe import (
    SerialPipeline,
    as_pickle_step,
    PlaceHolder,
)
from ezpyp.utils import (
    RepeatedStepError,
    SchemaConflictError,
    InitializationError,
    NotExecutedStepWarning,
    UnrecognizedStepWarning,
    SkippedStepWarning,
    FailedStepWarning,
)


def test_attachment(tmp_path):
    pipeline = SerialPipeline(tmp_path, "yikes")

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def foo():
        pass

    foo_step = foo()

    assert foo_step in pipeline.steps

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def bar():
        pass

    bar_step = bar()

    assert bar_step in pipeline.steps

    assert len(pipeline.steps) == 2

    # Initialization basically attempts to re-add duplicated step
    with pytest.raises(RepeatedStepError):
        bar()


def test_pipeline_with_no_dependencies(tmp_path):
    pipeline = SerialPipeline(tmp_path, "simple")

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def foo():
        return 2

    step_foo = foo()

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def bar(x):
        return x**2

    step_bar = bar(2)

    pipeline.organize_steps()

    assert len(pipeline.phases) == 1

    for step in pipeline.steps:
        assert step.get_status() == -1  # Not yet computed !

    pipeline.execute()

    for step, expected_result in zip((step_foo, step_bar), (2, 4)):
        assert step.get_status() == 0
        assert step.get_result() == expected_result


def test_pipeline_that_fails_with_no_dependencies(tmp_path):
    pipeline = SerialPipeline(tmp_path, "simple")

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def foo(crash=True):
        if crash:
            raise RuntimeError("Crash")
        else:
            return 0

    foo_step = foo()

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def bar():
        pass

    bar_step = bar()

    @as_pickle_step(pipeline=pipeline, depends_on=[foo_step])
    def baz():
        pass

    baz_step = baz()

    pipeline.execute()

    assert foo_step.get_status() == 1
    assert bar_step.get_status() == 0
    assert baz_step.get_status() == 2


def test_pipeline_with_dependencies(tmp_path):
    pipeline = SerialPipeline(tmp_path, "with_deps")

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def foo():
        return 2

    step_foo = foo()

    @as_pickle_step(pipeline=pipeline, depends_on=[step_foo])
    def bar():
        return 123

    step_bar = bar()

    pipeline.organize_steps()

    assert len(pipeline.phases) == 2

    for step in pipeline.steps:
        assert step.get_status() == -1

    pipeline.execute()

    for step, expected_result in zip((step_foo, step_bar), (2, 123)):
        assert step.get_status() == 0
        assert step.get_result() == expected_result


def test_pipeline_with_dependency_subs(tmp_path):
    pipeline = SerialPipeline(tmp_path, "with_subs")

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def base():
        return 2

    step_base = base()

    @as_pickle_step(pipeline=pipeline, depends_on=[step_base])
    def alpha(x):
        return x**2

    step_next = alpha(PlaceHolder(step_base))

    for step in pipeline.steps:
        assert step.get_status() == -1

    pipeline.execute()

    for step, expected_result in zip((step_base, step_next), (2, 4)):
        assert step.get_status() == 0
        assert step.get_result() == expected_result


def test_schema_initialization(tmp_path):
    pipeline = SerialPipeline(tmp_path, "schema")

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def a(x):
        return x**2

    a(3)

    with pytest.raises(InitializationError):
        pipeline.get_schema()

    pipeline.initialize_schema()

    assert pipeline.schema_path.exists()

    # Adding more steps after initialization is forbidden
    with pytest.raises(InitializationError):

        @as_pickle_step(pipeline=pipeline, depends_on=[])
        def b(x):
            return x**2

        b(3)

    # Second call to init does nothing
    pipeline.initialize_schema()


def test_incompatible_schemas(tmp_path):
    for ii in range(2):
        pipeline = SerialPipeline(tmp_path, "a")

        @as_pickle_step(pipeline=pipeline, depends_on=[])
        def a(x):
            return x**2

        a(1)

        if ii == 0:
            pipeline.initialize_schema()
            del pipeline, a

    pipeline = SerialPipeline(tmp_path, "a")

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def a(x):
        return x**2

    a(1)

    @as_pickle_step(pipeline=pipeline, depends_on=[])
    def b(x):
        return x**3

    b(12)

    with pytest.raises(SchemaConflictError):
        pipeline.initialize_schema()


def test_summary(tmp_path):
    pipeline = SerialPipeline(tmp_path, "simple")

    @as_pickle_step(pipeline)
    def x():
        return 1

    x()

    pipeline.execute()

    assert pipeline.summary_path.exists()  # TODO: do better...


def test_get_result(tmp_path):
    pipeline = SerialPipeline(tmp_path, "simple")

    @as_pickle_step(pipeline)
    def x():
        return 1

    x_step = x()

    @as_pickle_step(pipeline, depends_on=[x_step])
    def y(zeta):
        return zeta

    y_step = y(PlaceHolder(x_step))

    @as_pickle_step(pipeline, depends_on=[y_step])
    def z():
        raise RuntimeError("Whoa")

    z_step = z()

    @as_pickle_step(pipeline, depends_on=[z_step])
    def skip():
        pass

    skip()

    # Check that error is/isn't raised before/after exec
    with pytest.warns(NotExecutedStepWarning):
        pipeline.get_result("x")
    pipeline.execute()
    pipeline.get_result("x")

    # Check that undefined step warns
    with pytest.warns(UnrecognizedStepWarning):
        pipeline.get_result("w")

    # Check failed steps are detected
    with pytest.warns(FailedStepWarning):
        pipeline.get_result("z")

    # Check skipped steps are detected
    with pytest.warns(SkippedStepWarning):
        pipeline.get_result("skip")

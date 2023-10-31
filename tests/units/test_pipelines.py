import pytest

from ezpyp.pipe import SerialPipeline, as_pickle_step, RepeatedStepError


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

    crash_step = foo()

    pipeline.execute()

    assert crash_step.get_status() == 1


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

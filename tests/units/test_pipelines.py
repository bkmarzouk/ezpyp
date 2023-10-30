from ezpyp.pipe import SerialPipeline, as_pickle_step


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

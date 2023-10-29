import pytest
from ezpyp.pipe import (
    _Step,
    DillCache,
    PickleCache,
    NumpyCache,
    MissingDependency,
    DillStep,
    PickleStep,
    NumpyStep,
)
from pickle import PicklingError
from types import LambdaType
import numpy as np


class _SetupCacheTests:
    pass_objects: tuple = ()
    fail_objects: tuple = ()
    cache_method: DillCache | PickleCache | NumpyCache
    extension = None

    def test_cache_and_load_object(self, tmp_path):
        for idx, test_obj in enumerate(self.pass_objects):
            path = tmp_path / f"test_obj_{idx}"

            if self.extension is not None:
                path = path.with_suffix(f".{self.extension}")

            self.cache_method._cache_object(path, test_obj)

            assert path.exists()

            loaded = self.cache_method._load_object(path)

            if isinstance(test_obj, LambdaType):  # Lazy compare byte code
                assert test_obj.__code__.co_code == loaded.__code__.co_code
            elif isinstance(test_obj, np.ndarray):
                elementwise: np.ndarray = test_obj == loaded
                assert elementwise.all()
            else:
                assert self.cache_method._load_object(path) == test_obj

        for idx, test_obj in enumerate(self.fail_objects):
            path = tmp_path / f"test_obj_{idx}"

            with pytest.raises(PicklingError):
                self.cache_method._cache_object(path, test_obj)


class TestPickleCache(_SetupCacheTests):
    pass_objects = ([1, 2], "foo", ("bar", "pizza"))
    fail_objects = (lambda x: x**2,)
    cache_method = PickleCache


class TestDillCache(_SetupCacheTests):
    pass_objects = (
        [1, 2],
        "foo",
        ("bar", "pizza"),
        lambda x: x**2,
    )
    cache_method = DillCache


class TestNumpyCache(_SetupCacheTests):
    pass_objects = (
        np.arange(0, 100, 10),
        np.array((2, 2), dtype=float),
        np.ones(10) * 5,
    )
    cache_method = NumpyCache
    extension = "npy"


def test_step_equality(tmp_path):
    step = _Step(
        cache_location=tmp_path,
        name="test",
        args=[],
        kwargs={},
        function=lambda x: None,
        depends_on=[],
        extra_suffix="",
    )

    assert step == _Step(
        cache_location=tmp_path,
        name="test",
        args=[],
        kwargs={},
        function=lambda x: None,
        depends_on=[],
        extra_suffix="",
    )

    assert step != _Step(
        cache_location=tmp_path,
        name="test2",
        args=[],
        kwargs={},
        function=lambda x: None,
        depends_on=[],
        extra_suffix="",
    )


def test_step_get_path(tmp_path):
    step_name = "test"
    step = _Step(
        cache_location=tmp_path,
        name=step_name,
        args=[],
        kwargs={},
        function=lambda x: None,
        depends_on=[],
        extra_suffix="",
    )
    assert step._get_object_path("foo") == tmp_path / f"{step_name}.foo"
    assert step.get_results_path() == tmp_path / f"{step_name}.step"
    assert step.get_status_path() == tmp_path / f"{step_name}.status"


def test_step_get_path_npy(tmp_path):
    step_name = "test"
    npy_step = NumpyStep(
        cache_location=tmp_path,
        name=step_name,
        args=[],
        kwargs={},
        function=lambda x: None,
        depends_on=[],
    )
    assert npy_step.get_results_path() == tmp_path / f"{step_name}.step.npy"
    assert npy_step.get_status_path() == tmp_path / f"{step_name}.status.npy"


def quick_step(*args, **kwargs):
    return _Step(*args, **kwargs, extra_suffix="")


def test_check_ready(tmp_path):
    step_name = "test"

    for step_class in (quick_step, DillStep, PickleStep, NumpyStep):
        base_step = step_class(
            cache_location=tmp_path,
            name=step_name,
            args=[],
            kwargs={},
            function=lambda x: None,
            depends_on=[],
        )

        assert base_step.status == -1

        next_step = step_class(
            cache_location=tmp_path,
            name=step_name,
            args=[],
            kwargs={},
            function=lambda x: None,
            depends_on=[base_step],
        )

        # Error should be raised when status != 0
        with pytest.raises(MissingDependency):
            next_step.check_ready()

        # Overwrite initialised value to mimic completion
        base_step.status = 0
        next_step.check_ready()

        another_step = step_class(
            cache_location=tmp_path,
            name=step_name,
            args=[],
            kwargs={},
            function=lambda x: None,
            depends_on=[next_step, base_step],
        )

        # Partially completed steps should also be invalid
        with pytest.raises(MissingDependency):
            another_step.check_ready()


def simple_function(x):
    return x


def function_w_kwargs(x, y=None):
    if y is not None:
        return x + y
    return x


def function_w_npy(x_start, x_end, x_n, amplitude, frequency, shift):
    xspace = np.linspace(x_start, x_end, x_n)
    return amplitude * np.cos(frequency * xspace + shift)


cache_and_load_test_data = [
    (simple_function, [123], {}, 123),
    (function_w_kwargs, [123], {"y": 1}, 124),
    (
        function_w_npy,
        [0, np.pi, 2, 10, 1, 0],
        {},
        np.array([10, -10], dtype=float),
    ),
]


@pytest.mark.parametrize(
    "function,args,kwargs,result", cache_and_load_test_data
)
def test_cache_and_load_result(tmp_path, function, args, kwargs, result):
    step_names = ("dill", "pickle", "numpy")
    step_constructors = (DillStep, PickleStep, NumpyStep)

    for step_class, step_name in zip(step_constructors, step_names):
        step = step_class(
            cache_location=tmp_path,
            name=step_name,
            args=args,
            kwargs=kwargs,
            function=function,
            depends_on=[],
        )

        # No dependencies, so should be fine!
        step.check_ready()

        assert step.status == -1
        # Compute and cache
        direct_calculation = step.get_result()
        cached_calculation = step.get_result()
        if isinstance(result, np.ndarray):
            assert (direct_calculation == cached_calculation).all()
            assert (direct_calculation == result).all()
            assert (cached_calculation == result).all()
        else:
            assert direct_calculation == cached_calculation == result
        assert step.status == 0

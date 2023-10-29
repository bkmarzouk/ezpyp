import pytest
from ezpyp.pipe import DillCache, PickleCache, NumpyCache
from pickle import PicklingError
from types import LambdaType
import numpy as np


class _SetupCacheTests:
    pass_objects: tuple = ()
    fail_objects: tuple = ()
    cache_method: DillCache | PickleCache | NumpyCache
    extension = None

    def test_cache_and_load_object(self, tmp_path):
        print("HELLO", self.cache_method)

        for idx, test_obj in enumerate(self.pass_objects):
            print("TESTING", test_obj)

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

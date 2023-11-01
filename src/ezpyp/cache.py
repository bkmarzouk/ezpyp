import pickle
from pathlib import Path
from typing import Any

import dill
import numpy as np


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

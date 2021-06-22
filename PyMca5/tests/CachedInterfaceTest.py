import unittest
import numpy
from collections import Counter
from PyMca5.PyMcaMath.fitting.CachedInterface import CachedInterface
from PyMca5.PyMcaMath.fitting.CachedInterface import cached_property


class Cached(CachedInterface):
    def __init__(self):
        super().__init__()
        self._cfg = {"var1": 1, "var2": 2}
        self.reset_counters()

    def reset_counters(self):
        self.get_counter = Counter()
        self.set_counter = Counter()

    @cached_property
    def var1(self):
        self.get_counter["var1"] += 1
        return self._cfg.get("var1")

    @var1.setter
    def var1(self, value):
        self.set_counter["var1"] += 1
        self._cfg["var1"] = value

    @cached_property
    def var2(self):
        self.get_counter["var2"] += 1
        return self._cfg.get("var2")

    @var2.setter
    def var2(self, value):
        self.set_counter["var2"] += 1
        self._cfg["var2"] = value

    def _property_cache_index(self, name):
        if name == "var1":
            return 0
        else:
            return 1

    def _property_cache_key(self, **_):
        return None

    def _create_empty_cache(self, key, **_):
        return numpy.zeros(2, dtype=float)


class testCachedInterface(unittest.TestCase):
    def setUp(self):
        self.cached_object = Cached()

    def _assertGetSetCount(self, name, getcount, setcount):
        self.assertEqual(self.cached_object.get_counter[name], getcount)
        self.assertEqual(self.cached_object.set_counter[name], setcount)

    def _assertVarValues(self, v1, v2):
        self.assertEqual(self.cached_object.var1, v1)
        self.assertEqual(self.cached_object.var2, v2)

    def test_get_without_caching(self):
        for i in range(5):
            self.assertEqual(self.cached_object.var1, 1)
            self.assertEqual(self.cached_object.var2, 2)
            self._assertGetSetCount("var1", i + 1, 0)
            self._assertGetSetCount("var2", i + 1, 0)

    def test_set_without_caching(self):
        for i in range(5):
            self.cached_object.var1 = 100
            self.cached_object.var2 = 200
            self._assertGetSetCount("var1", 0, i + 1)
            self._assertGetSetCount("var2", 0, i + 1)
        self._assertVarValues(100, 200)

    def test_get_with_caching(self):
        with self.cached_object.propertyCachingContext() as cache:
            for i in range(5):
                self.assertEqual(self.cached_object.var1, 1)
                self.assertEqual(self.cached_object.var2, 2)
            self.assertEqual(cache.tolist(), [1, 2])
        self._assertGetSetCount("var1", 1, 0)
        self._assertGetSetCount("var2", 1, 0)

    def test_set_with_caching(self):
        with self.cached_object.propertyCachingContext() as cache:
            for i in range(5):
                self.cached_object.var1 = 100
                self.cached_object.var2 = 200
                self.assertEqual(self.cached_object.var1, 100)
                self.assertEqual(self.cached_object.var2, 200)
                self.assertEqual(cache.tolist(), [100, 200])
        self._assertGetSetCount("var1", 1, 0)
        self._assertGetSetCount("var2", 1, 0)
        self._assertVarValues(1, 2)

    def test_set_with_persistent_caching(self):
        with self.cached_object.propertyCachingContext(persist=True) as cache:
            for i in range(5):
                self.cached_object.var1 = 100
                self.cached_object.var2 = 200
                self.assertEqual(self.cached_object.var1, 100)
                self.assertEqual(self.cached_object.var2, 200)
                self.assertEqual(cache.tolist(), [100, 200])
        self._assertGetSetCount("var1", 1, 1)
        self._assertGetSetCount("var2", 1, 1)
        self._assertVarValues(100, 200)

    def test_start_cache(self):
        with self.cached_object.propertyCachingContext(start_cache=[100, 200]) as cache:
            self.assertEqual(cache, [100, 200])
            self.assertEqual(self.cached_object.var1, 100)
            self.assertEqual(self.cached_object.var2, 200)
        self._assertGetSetCount("var1", 0, 0)
        self._assertGetSetCount("var2", 0, 0)
        self._assertVarValues(1, 2)

    def test_persistent_start_cache(self):
        with self.cached_object.propertyCachingContext(
            start_cache=[100, 200], persist=True
        ) as cache:
            self.assertEqual(cache, [100, 200])
        self._assertGetSetCount("var1", 0, 1)
        self._assertGetSetCount("var2", 0, 1)
        self._assertVarValues(100, 200)

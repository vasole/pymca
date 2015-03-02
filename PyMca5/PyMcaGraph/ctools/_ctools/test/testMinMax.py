# -*- coding: utf-8 -*-

"""Tests for C minMax"""


# import ######################################################################

import numpy as np
import time
try:
    import unittest
except ImportError:
    import unittest2 as unittest

from PyMca5.PyMcaGraph import ctools


# TestMinMax ##################################################################

class TestMinMax(unittest.TestCase):
    """Test minMax in C.

    Test with different: data types, sizes.
    """

    # Array data types to test
    SIGNED_DTYPES = (np.float32, np.float64,
                     np.int8, np.int16,
                     np.int32, np.int64)
    UNSIGNED_DTYPES = (np.uint8, np.uint16,
                       np.uint32, np.uint64)
    DTYPES = SIGNED_DTYPES + UNSIGNED_DTYPES

    # Array sizes to test
    SIZES = 10, 256, 1024, 2048, 4096 #, 4096 ** 2, 8192 ** 2

    def _log(self, *args):
        """Logging used by test for debugging."""
        pass
        # print(args)

    def _testMinMax(self, data):
        """Single test C minMax vs Numpy min/max."""
        startTime = time.time()
        min_, max_ = ctools.minMax(data)
        duration = time.time() - startTime

        startTime = time.time()
        minNumpy, maxNumpy = data.min(), data.max()
        durationNumpy = time.time() - startTime
        self._log(data.dtype, data.size, 'duration (s):', duration, 'Numpy/C:',
                  durationNumpy / duration)
        self.assertEqual(min_, minNumpy)
        self.assertEqual(max_, maxNumpy)

    def testMinMax(self):
        """Test C minMax vs Numpy min/max for different data types and sizes.
        """
        self._log("testMinMax")
        for size in self.SIZES:
            for dtype in self.DTYPES:
                data = np.arange(size, dtype=dtype)
                self._testMinMax(data)

                data = np.arange(size, 0, -1, dtype=dtype)
                self._testMinMax(data)

    @staticmethod
    def _minPos(data):
        posValue = np.take(data, np.nonzero(data > 0))
        if posValue.size != 0:
            return posValue.min()
        else:
            return None  # if no value above 0

    def _testMinMinPositiveMax(self, data):
        """Single test C minMax and min positive vs Numpy min/max."""
        startTime = time.time()
        min_, minPositive, max_ = ctools.minMax(data, minPositive=True)
        duration = time.time() - startTime

        startTime = time.time()
        try:
            minNumpy, maxNumpy = data.min(), data.max()
        except ValueError:
            minNumpy, maxNumpy = None, None
        minPositiveNumpy = self._minPos(data)
        durationNumpy = time.time() - startTime

        self._log(data.dtype, data.size, 'duration (s):', duration, 'Numpy/C:',
                  durationNumpy / duration)

        self.assertEqual(min_, minNumpy)
        self.assertEqual(minPositive, minPositiveNumpy)
        self.assertEqual(max_, maxNumpy)

    def testMinMinPosMax(self):
        """Test C minMax and min positive vs Numpy.
        """
        self._log("testMinMinPosMax")
        for size in self.SIZES:
            for dtype in self.DTYPES:
                # Increasing data
                data = np.arange(size, dtype=dtype)
                self._testMinMinPositiveMax(data)

                # Decreasing data
                data = np.arange(size, 0, -1, dtype=dtype)
                self._testMinMinPositiveMax(data)

    def testMinMinPosMaxSomeNegative(self):
        """Test C minMax and min positive vs Numpy with some negative data.
        """
        self._log("testMinMinPosMaxAllNegative")
        for size in self.SIZES:
            for dtype in self.SIGNED_DTYPES:
                # All negative data
                data = np.arange(-int(size/2.), 0, dtype=dtype)
                self._testMinMinPositiveMax(data)

    def testMinMinPosMaxAllNegative(self):
        """Test C minMax and min positive vs Numpy with all negative data.
        """
        self._log("testMinMinPosMaxAllNegative")
        for size in self.SIZES:
            for dtype in self.SIGNED_DTYPES:
                # All negative data
                data = np.arange(-size, 0, dtype=dtype)
                self._testMinMinPositiveMax(data)

    def testMinMaxNoData(self):
        """Test C minMax and min positive with no data.
        """
        self._log("testMinMaxNoData")
        for dtype in self.DTYPES:
            # No data
            data = np.array((), dtype=dtype)
            with self.assertRaises(ValueError):
                result = ctools.minMax(data, minPositive=False)

            with self.assertRaises(ValueError):
                result = ctools.minMax(data, minPositive=True)


# main ########################################################################

if __name__ == '__main__':
    import sys

    unittest.main(argv=sys.argv[:])

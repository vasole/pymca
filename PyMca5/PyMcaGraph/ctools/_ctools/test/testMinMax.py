# -*- coding: utf-8 -*-

"""Tests for C minMax"""


# import ######################################################################

import math
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
    FLOATING_DTYPES = np.float16, np.float32, np.float64
    SIGNED_DTYPES = FLOATING_DTYPES + (np.int8, np.int16, np.int32, np.int64)
    UNSIGNED_DTYPES = np.uint8, np.uint16, np.uint32, np.uint64
    DTYPES = SIGNED_DTYPES + UNSIGNED_DTYPES

    # Array sizes to test
    SIZES = 10, 256, 1024, 2048, 4096  # , 4096 ** 2, 8192 ** 2

    def _log(self, *args):
        """Logging used by test for debugging."""
        # pass
        print(args)

    @staticmethod
    def _minPos(data):
        posValue = np.take(data, np.nonzero(data > 0))
        if posValue.size != 0:
            return posValue.min()
        else:
            return None  # if no value above 0

    def _testMinMaxVsNumpy(self, data, minPos=False):
        """Single test C minMax and min positive vs Numpy min/max."""
        startTime = time.time()
        if minPos:
            min_, minPositive, max_ = ctools.minMax(data, minPositive=True)
        else:
            min_, max_ = ctools.minMax(data, minPositive=False)
        duration = time.time() - startTime

        startTime = time.time()
        try:
            minNumpy, maxNumpy = np.nanmin(data), np.nanmax(data)
        except ValueError:
            minNumpy, maxNumpy = None, None
        if minPos:
            minPositiveNumpy = self._minPos(data)
        durationNumpy = time.time() - startTime

        self._log(data.dtype, data.size, 'duration (s):', duration, 'Numpy/C:',
                  durationNumpy / duration)

        self.assertEqual(min_, minNumpy)
        if minPos:
            self.assertEqual(minPositive, minPositiveNumpy)
        self.assertEqual(max_, maxNumpy)

    # @unittest.skip("")
    def testMinMaxOnly(self):
        """Test C minMax vs Numpy min/max for different data types and sizes.
        """
        self._log("testMinMax")
        for size in self.SIZES:
            for dtype in self.DTYPES:
                data = np.arange(size, dtype=dtype)
                self._testMinMaxVsNumpy(data, False)

                data = np.arange(size, 0, -1, dtype=dtype)
                self._testMinMaxVsNumpy(data, False)

    # @unittest.skip("")
    def testMinMax(self):
        """Test C minMax and min positive vs Numpy.
        """
        self._log("testMinMinPosMax")
        for size in self.SIZES:
            for dtype in self.DTYPES:
                # Increasing data
                data = np.arange(size, dtype=dtype)
                self._testMinMaxVsNumpy(data)

                # Decreasing data
                data = np.arange(size, 0, -1, dtype=dtype)
                self._testMinMaxVsNumpy(data)

    # @unittest.skip("")
    def testMinMinPosMaxSomeNegative(self):
        """Test C minMax and min positive vs Numpy with some negative data.
        """
        self._log("testMinMinPosMaxAllNegative")
        for size in self.SIZES:
            for dtype in self.SIGNED_DTYPES:
                # Some negative data
                data = np.arange(-int(size/2.), size, dtype=dtype)
                self._testMinMaxVsNumpy(data)

    # @unittest.skip("")
    def testMinMinPosMaxAllNegative(self):
        """Test C minMax and min positive vs Numpy with all negative data.
        """
        self._log("testMinMinPosMaxAllNegative")
        for size in self.SIZES:
            for dtype in self.SIGNED_DTYPES:
                # All negative data
                data = np.arange(-size, 0, dtype=dtype)
                self._testMinMaxVsNumpy(data)

    # @unittest.skip("")
    def testMinMaxNoData(self):
        """Test C minMax and min positive with no data.
        """
        self._log("testMinMaxNoData")
        for dtype in self.DTYPES:
            # No data
            data = np.array((), dtype=dtype)
            with self.assertRaises(ValueError):
                ctools.minMax(data, minPositive=False)

            with self.assertRaises(ValueError):
                ctools.minMax(data, minPositive=True)

    # @unittest.skip("")
    def testMinMaxNan(self):
        """Test C minMax and min positive with NaN.
        """
        self._log("testMinMaxNan")

        for dtype in self.FLOATING_DTYPES:
            # All NaN
            data = np.array((float('nan'), float('nan')), dtype=dtype)
            min_, minPositive, max_ = ctools.minMax(data, minPositive=True)
            self.assertTrue(math.isnan(min_))
            self.assertEqual(minPositive, None)
            self.assertTrue(math.isnan(max_))

            # NaN first and positive
            data = np.array((float('nan'), 1.0), dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # NaN first and negative
            data = np.array((float('nan'), -1.0), dtype=dtype)
            self._testMinMaxVsNumpy(data)
 
            # NaN last and positive
            data = np.array((1.0, 2.0, float('nan')), dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # NaN last and negative
            data = np.array((-1.0, -2.0, float('nan')), dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # Some NaN
            data = np.array((1.0, float('nan'), -1.0), dtype=dtype)
            self._testMinMaxVsNumpy(data)

    # @unittest.skip("")
    def testMinMaxInf(self):
        """Test C minMax and min positive with Inf.
        """
        self._log("testMinMaxInf")

        for dtype in self.FLOATING_DTYPES:
            # All Positive Inf
            data = np.array((float('inf'), float('inf')), dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # All Negative Inf
            data = np.array((float('-inf'), float('-inf')), dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # Positive and negative Inf
            data = np.array((float('inf'), float('-inf')), dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # Positive and negative Inf and NaN first
            data = np.array((float('nan'), float('inf'), float('-inf')),
                            dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # Positive and negative Inf and NaN last
            data = np.array((float('inf'), float('-inf'), float('nan')),
                            dtype=dtype)
            self._testMinMaxVsNumpy(data)

            # Positive and negative Inf and NaN last
            data = np.array((float('inf'), float('-inf'), float('nan')),
                            dtype=dtype)
            self._testMinMaxVsNumpy(data)


# main ########################################################################

if __name__ == '__main__':
    import sys

    unittest.main(argv=sys.argv[:])

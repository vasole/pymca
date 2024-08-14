# -*- coding: utf-8 -*-
#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
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
        pass
        # print(args)

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

        self._log(data.dtype, data.size, 'duration C (s):', duration,
                  'duration Numpy (s):', durationNumpy)

        self.assertEqual(min_, minNumpy)
        if minPos:
            self.assertEqual(minPositive, minPositiveNumpy)
        self.assertEqual(max_, maxNumpy)

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

    def testMinMinPosMaxSomeNegative(self):
        """Test C minMax and min positive vs Numpy with some negative data.
        """
        self._log("testMinMinPosMaxAllNegative")
        for size in self.SIZES:
            for dtype in self.SIGNED_DTYPES:
                # Some negative data
                data = np.arange(-int(size/2.), size, dtype=dtype)
                self._testMinMaxVsNumpy(data)

    def testMinMinPosMaxAllNegative(self):
        """Test C minMax and min positive vs Numpy with all negative data.
        """
        self._log("testMinMinPosMaxAllNegative")
        for size in self.SIZES:
            for dtype in self.SIGNED_DTYPES:
                # All negative data
                data = np.arange(-size, 0, dtype=dtype)
                self._testMinMaxVsNumpy(data)

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

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
    DTYPES = (np.float32, np.float64,
              np.int8, np.uint8,
              np.int16, np.uint16,
              np.int32, np.uint32,
              np.int64, np.uint64)

    # Array sizes to test
    SIZES = 10, 256, 1024, 2048, 4096, 4096 ** 2, 8192 ** 2

    def _log(self, *args):
        """Logging used by test for debugging."""
        pass
        # print(*args)

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
        for size in self.SIZES:
            for dtype in self.DTYPES:
                data = np.arange(size, dtype=dtype)
                self._testMinMax(data)

                data = np.arange(size, 0, -1, dtype=dtype)
                self._testMinMax(data)


# main ########################################################################

if __name__ == '__main__':
    import sys

    unittest.main(argv=sys.argv[:])

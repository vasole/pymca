# -*- coding: utf-8 -*-

# import ######################################################################

import math
import numpy as np
import random
import time
import struct
import sys
try:
    import unittest
except ImportError:
    import unittest2 as unittest

from PyMca5.PyMcaGraph import ctools


# TODOs:
# benchmark perf

# common ######################################################################

class TestFastLog10(unittest.TestCase):
    """Test C fastLog10."""

    @staticmethod
    def _log(*args):
        """Logging used by test for debugging."""
        pass
        # print(args)

    def testReturnDefined(self):
        """Test specific values."""
        # Test cases as (value, log10(value))
        testCases = (
            (0.0, float('-inf')),
            (1.0, 0.0),
            (float('inf'), float('inf')),
        )

        for value, refLogValue in testCases:
            logValue = ctools.fastLog10(value)
            self.assertEqual(logValue, refLogValue)

    def testReturnNan(self):
        """Test values for which log10(value) returns NaN.
        
        Test: NaN, -Inf, and a few negative values.
        """
        testValues = (
            float('nan'),
            float('-inf'),
            -1.0,
            - sys.float_info.max,
            - sys.float_info.min
        )

        for value in testValues:
            logValue = ctools.fastLog10(value)
            self.assertTrue(math.isnan(logValue))

    @staticmethod
    def _randFloat():
        """Returns a random float truly over the full range of 64-bits floats.

        Can produce Nan, +/- inf.
        """
        return struct.unpack('d', struct.pack('L', random.getrandbits(64)))[0]

    def _randPosFloat(self):
        """Returns a strictly positive random float."""
        value = self._randFloat()
        while value < 0.0 or math.isinf(value) or math.isnan(value):
            value = self._randFloat()
        return value

    # @unittest.skip("Not for reproductible tests")
    def testRandomPositive(self):
        """Test with strictly positive random values."""
        self._log("testRandomPositive")

        # Create data set
        nbData = 10 ** 6
        values = [self._randPosFloat() for i in range(nbData)]
        dataRange = min(values), max(values)
        self._log("Nb data:", nbData, "in range:", dataRange)

        # Compute log10
        logValues = map(ctools.fastLog10, values)
        refLogValues = map(math.log10, values)

        # Comparison
        errors = map(lambda a, b: math.fabs(a - b), logValues, refLogValues)
        bigErrors = filter(lambda a : a > 0.0001, errors)
        self.assertEqual(len(bigErrors), 0)

        self._log("Errors > 0.0001", bigErrors)
        self._log("Max Error:", max(errors))


# main ########################################################################

if __name__ == '__main__':
    import sys

    unittest.main(argv=sys.argv[:])

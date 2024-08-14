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
        return struct.unpack('d', struct.pack('Q', random.getrandbits(64)))[0]

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
        errors = list(map(lambda a, b: math.fabs(a - b),
                      logValues, refLogValues))
        bigErrors = list(filter(lambda a : a > 0.00011, errors))

        self._log("Nb errors > 0.00011", len(bigErrors))
        self._log("Max Error:", max(errors))

        self.assertEqual(len(bigErrors), 0)


# main ########################################################################

if __name__ == '__main__':
    import sys

    unittest.main(argv=sys.argv[:])

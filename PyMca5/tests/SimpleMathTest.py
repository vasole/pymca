#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2023 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V. Armando Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import unittest
import os
import sys
import numpy

class testSimpleMath(unittest.TestCase):
    def _testDerivativeHelper(self, option=None):
        from PyMca5.PyMcaMath import SimpleMath
        x = numpy.arange(100.)*0.25
        y = x*x + 2 * x
        a = SimpleMath.SimpleMath()
        xplot, yprime = a.derivate(x, y, option=option)
        for i in range(yprime.size - 10):
            self.assertTrue(numpy.abs(yprime[i] - 2 * xplot[i] - 2 ) < 1.0e-4,
                        "Error 1 at %d yprime = %f x = %f" % (i, yprime[i], xplot[i]))
        x = -x
        y = x*x + 2 * x
        xplot, yprime = a.derivate(x, y)
        for i in range(yprime.size - 10):
            self.assertTrue(numpy.abs(yprime[i] - 2 * xplot[i] - 2 ) < 1.0e-4,
                        "Error 2 at %d yprime = %f x = %f" % (i, yprime[i], xplot[i]))

    def testDerivativeSinglePoint(self):
        self._testDerivativeHelper()

    def testDerivativeSavitzkyGolay(self):
        self._testDerivativeHelper(option="SG smoothed 3 point")


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testSimpleMath))
    else:
        # use a predefined order
        testSuite.addTest(testSimpleMath("testDerivativeSinglePoint"))
        testSuite.addTest(testSimpleMath("testDerivativeSavitzkyGolay"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()

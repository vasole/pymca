#/*##########################################################################
# Copyright (C) 2004 - 2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
import unittest
import os
import numpy    

class testGefit(unittest.TestCase):
    def setUp(self):
        """
        import the module
        """
        try:
            from PyMca5 import Gefit
            self.gefit = Gefit
        except:
            self.gefit = None

    def gaussianPlusLinearBackground(self, param, t):
        dummy = 2.3548200450309493 * (t - param[3])/ param[4]
        return param[0] + param[1] * t +\
               param[2] * numpy.exp(-0.5 * dummy * dummy)

    def testGefitImport(self):
        self.assertTrue(self.gefit is not None)

    def testGefitLeastSquares(self):
        self.testGefitImport()
        x = numpy.arange(500.)
        originalParameters = numpy.array([10.5, 2, 1000.0, 200., 100],
                                         numpy.float)
        fitFunction = self.gaussianPlusLinearBackground
        y = fitFunction(originalParameters, x)

        startingParameters = [0.0 ,1.0,900.0, 150., 90]
        fittedpar, chisq, sigmapar =self.gefit.LeastSquaresFit(fitFunction,
                                                     startingParameters,
                                                     xdata=x,
                                                     ydata=y,
                                                     sigmadata=None)
        for i in range(len(originalParameters)):
            self.assertTrue(abs(fittedpar[i] - originalParameters[i]) < 0.01)

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testGefit))
    else:
        # use a predefined order
        testSuite.addTest(testGefit("testGefitImport"))
        testSuite.addTest(testGefit("testGefitLeastSquares"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()

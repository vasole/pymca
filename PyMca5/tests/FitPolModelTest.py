# /*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
__author__ = "Wout De Nolf"
__contact__ = "wout.de_nolf@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import unittest
import numpy
from PyMca5.PyMcaMath.fitting.model import PolynomialModels


class testFitPolModel(unittest.TestCase):
    def setUp(self):
        self.random_state = numpy.random.RandomState(seed=0)

    def testLinearPol(self):
        model = PolynomialModels.LinearPolynomialModel()
        fitmodel = PolynomialModels.LinearPolynomialModel()
        model.xdata = fitmodel.xdata = numpy.linspace(0, 100, 100)

        for degree in [0, 1, 5]:
            with self.subTest(degree=degree):
                model.degree = degree
                fitmodel.degree = degree
                ncoeff = degree + 1
                expected = self.random_state.uniform(low=-5, high=5, size=ncoeff)
                model.coefficients = expected
                actual = model.get_parameter_values()
                numpy.testing.assert_array_equal(actual, expected)

                names = model.get_parameter_group_names()
                expected_names = "fitmodel_coefficients",
                self.assertEqual(names, expected_names)

                fitmodel.ydata = model.yfullmodel
                numpy.testing.assert_array_equal(fitmodel.ydata, model.yfullmodel)
                numpy.testing.assert_array_equal(fitmodel.yfitdata, model.yfitmodel)
                numpy.testing.assert_array_equal(model.yfitmodel, model.yfullmodel)

                for linear in [True, False]:
                    with self.subTest(degree=degree, linear=linear):
                        fitmodel.linear = linear
                        fitmodel.coefficients = numpy.zeros_like(expected)
                        self.assertEqual(fitmodel.degree, degree)
                        result = fitmodel.fit()["parameters"]
                        numpy.testing.assert_allclose(result, expected, rtol=1e-4)

    def testExpPol(self):
        model = PolynomialModels.ExponentialPolynomialModel()
        fitmodel = PolynomialModels.ExponentialPolynomialModel()
        model.xdata = fitmodel.xdata = numpy.linspace(-0.5, 0.5, 100)

        for degree in [0, 1, 5]:
            with self.subTest(degree=degree):
                model.degree = degree
                fitmodel.degree = degree
                ncoeff = degree + 1
                expected = self.random_state.uniform(low=-5, high=5, size=ncoeff)
                model.coefficients = expected
                expected[0] = numpy.log(expected[0])
                actual = model.get_parameter_values()
                numpy.testing.assert_array_equal(actual, expected)

                fitmodel.ydata = model.yfullmodel
                numpy.testing.assert_array_equal(fitmodel.ydata, model.yfullmodel)
                numpy.testing.assert_allclose(fitmodel.yfitdata, model.yfitmodel)
                numpy.testing.assert_allclose(
                    model.yfitmodel, numpy.log(model.yfullmodel)
                )

                for linear in [True, False]:
                    with self.subTest(degree=degree, linear=linear):
                        fitmodel.linear = linear
                        fitmodel.coefficients = numpy.zeros_like(expected)
                        if not linear:
                            fitmodel.coefficients[0] = 0.1
                        self.assertEqual(fitmodel.degree, degree)
                        result = fitmodel.fit()["parameters"]
                        numpy.testing.assert_allclose(result, expected)


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testFitPolModel))
    else:
        # use a predefined order
        testSuite.addTest(testFitPolModel("testLinearPol"))
        testSuite.addTest(testFitPolModel("testExpPol"))
    return testSuite


def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))


if __name__ == "__main__":
    test()

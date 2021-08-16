import unittest
import numpy
from PyMca5.PyMcaMath.fitting.model import PolynomialModels
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterType
from PyMca5.PyMcaMath.fitting.model.ParameterModel import AllParameterTypes


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
                expected_names = ("fitmodel_coefficients",)
                self.assertEqual(names, expected_names)

                fitmodel.ydata = model.yfullmodel
                numpy.testing.assert_array_equal(fitmodel.ydata, model.yfullmodel)
                numpy.testing.assert_array_equal(fitmodel.yfitdata, model.yfitmodel)
                numpy.testing.assert_array_equal(model.yfitmodel, model.yfullmodel)

                for parameter_types in [
                    ParameterType.independent_linear,
                    AllParameterTypes,
                ]:
                    with self.subTest(degree=degree, parameter_types=parameter_types):
                        fitmodel.parameter_types = parameter_types
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

                for parameter_types in [
                    ParameterType.independent_linear,
                    AllParameterTypes,
                ]:
                    with self.subTest(degree=degree, parameter_types=parameter_types):
                        fitmodel.parameter_types = parameter_types
                        fitmodel.coefficients = numpy.zeros_like(expected)
                        if parameter_types == AllParameterTypes:
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

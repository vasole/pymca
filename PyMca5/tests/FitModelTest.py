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
from PyMca5.tests import SimpleModel


def with_model(nmodels):
    def inner1(method):
        def inner2(self, *args, **kw):
            self.create_model(nmodels=nmodels)
            result = method(self, *args, **kw)
            self.validate_model()
            return result

        return inner2

    return inner1


class testFitModel(unittest.TestCase):
    def setUp(self):
        self.random_state = numpy.random.RandomState(seed=100)

    def create_model(self, nmodels):
        self.nmodels = nmodels
        self.is_concat = nmodels != 1
        if nmodels == 1:
            self.fitmodel = SimpleModel.SimpleModel()
        else:
            self.fitmodel = SimpleModel.SimpleConcatModel(ndetectors=nmodels)
        self.assertTrue(not self.fitmodel.linear)
        self.init_random()
        ydata = self.fitmodel.yfullmodel.copy()
        self.fitmodel.ydata = ydata
        numpy.testing.assert_array_equal(self.fitmodel.ydata, ydata)
        numpy.testing.assert_array_equal(self.fitmodel.yfullmodel, ydata)
        numpy.testing.assert_allclose(self.fitmodel.yfitmodel, ydata - 10)
        self.validate_model()

    def init_random(self, **kw):
        if self.is_concat:
            for model in self.fitmodel._models:
                self._init_random(model, **kw)
            self.fitmodel.shared_attributes = self.fitmodel.shared_attributes
        else:
            self._init_random(self.fitmodel, **kw)

    def _init_random(self, model, npeaks=10, nchannels=2048, border=0.1):
        """Peaks close to the border will cause the nlls to fail"""
        self.npeaks = npeaks  # concentrations
        self.nshapeparams = 4  # zero, gain, wzero, wgain
        model.xdata_raw = numpy.arange(nchannels)
        model.ydata_raw = numpy.full(nchannels, numpy.nan)
        model.ybkg = 10
        model.xmin = self.random_state.randint(low=0, high=10)
        model.xmax = self.random_state.randint(low=nchannels - 10, high=nchannels)
        model.zero = self.random_state.uniform(low=1, high=1.5)
        model.gain = self.random_state.uniform(low=10e-3, high=11e-3)
        a = model.zero
        b = model.zero + model.gain * nchannels
        border = border * (b - a)
        a += border
        b -= border
        model.positions = numpy.linspace(a, b, npeaks)
        model.wzero = self.random_state.uniform(low=0.0, high=0.01)
        model.wgain = self.random_state.uniform(low=0.05, high=0.1)
        model.concentrations = self.random_state.uniform(low=0.5, high=1, size=npeaks)
        model.efficiency = self.random_state.uniform(low=5000, high=6000, size=npeaks)

    def modify_random(self, only_linear=False):
        self._modify_random(only_linear=only_linear)
        self.validate_model()
        # self.plot()

    def _modify_random(self, only_linear=False):
        porg = self.fitmodel.parameters.copy()
        plinorg = self.fitmodel.linear_parameters.copy()
        if only_linear:
            plin = self.fitmodel.linear_parameters
            plin *= self.random_state.uniform(0.5, 0.8, len(plin))
            self.fitmodel.linear_parameters = plin
            numpy.testing.assert_array_equal(self.fitmodel.linear_parameters, plin)
            p = self.fitmodel.parameters
        else:
            p = self.fitmodel.parameters
            p *= self.random_state.uniform(0.95, 1, len(p))
            self.fitmodel.parameters = p
            numpy.testing.assert_array_equal(self.fitmodel.parameters, p)
            plin = self.fitmodel.linear_parameters

        for param_idx, param_name in enumerate(self.fitmodel.parameter_names):
            if "concentration" in param_name or not only_linear:
                self.assertNotEqual(p[param_idx], porg[param_idx], msg=param_name)
            else:
                self.assertEqual(p[param_idx], porg[param_idx], msg=param_name)

        for param_idx, param_name in enumerate(self.fitmodel.linear_parameter_names):
            self.assertNotEqual(plin[param_idx], plinorg[param_idx], msg=param_name)

        return p

    def validate_model(self):
        self._validate_model(self.fitmodel, self.is_concat)
        if self.is_concat:
            for model in self.fitmodel._models:
                self._validate_model(model, False)

    def _validate_model(self, model, is_concat):
        if not is_concat:
            # Alphabetic order
            expected = ["concentrations", "gain", "wgain", "wzero", "zero"]
            self.assertEqual(model.parameter_group_names, expected)
            expected = ["concentrations"]
            self.assertEqual(model.linear_parameter_group_names, expected)
        self.assertTrue(not model._excluded_parameters)
        self.assertTrue(not model._included_parameters)
        self.assertEqual(model.ndata, len(model.xdata))
        self.assertEqual(model.nparameters, len(model.parameters))
        self.assertEqual(model.nlinear_parameters, len(model.linear_parameters))

        arr1 = model.evaluate_fullmodel()
        arr2 = model.evaluate_linear_fullmodel()
        arr3 = model.yfullmodel
        numpy.testing.assert_allclose(arr1, arr2)
        numpy.testing.assert_allclose(arr1, arr3)

        arr1 = model.evaluate_fitmodel()
        arr2 = model.evaluate_linear_fitmodel()
        arr3 = model.yfitmodel
        arr4 = sum(model.linear_decomposition_fitmodel())
        numpy.testing.assert_allclose(arr1, arr2)
        numpy.testing.assert_allclose(arr1, arr3)
        numpy.testing.assert_allclose(arr1, arr4)

        # Alphabetic order
        nonlin_names = ["gain", "wgain", "wzero", "zero"]
        lin_names = ["concentrations" + str(i) for i in range(self.npeaks)]
        names = lin_names + nonlin_names
        if is_concat:
            model.validate_shared_attributes()
            self.assertEqual(model.nshared_parameters, self.npeaks)
            self.assertEqual(model.nshared_linear_parameters, self.npeaks)
            nmodels = model.nmodels
            names = lin_names + [
                name + str(i) for i in range(nmodels) for name in nonlin_names
            ]
            n = self.npeaks + self.nshapeparams * nmodels
            self.assertEqual(model.nparameters, n)
            self.assertEqual(model.nlinear_parameters, self.npeaks)
            self.assertEqual(model.parameter_names, names)
            self.assertEqual(model.linear_parameter_names, lin_names)
        else:
            self.assertEqual(model.nparameters, self.npeaks + self.nshapeparams)
            self.assertEqual(model.nlinear_parameters, self.npeaks)
            self.assertEqual(model.parameter_names, names)
            self.assertEqual(model.linear_parameter_names, lin_names)

    def plot(self):
        import matplotlib.pyplot as plt

        m = self.fitmodel
        derivatives = m.derivatives()
        names = m.parameter_names
        plt.figure()
        plt.plot(m.ydata, label="data")
        plt.plot(m.yfitmodel, label="model")
        plt.legend()
        plt.figure()
        for y, name in zip(derivatives, names):
            plt.plot(y, label=name)
        plt.title("Derivatives")
        plt.legend()
        plt.show()

    @with_model(1)
    def testLinearFit(self):
        self._testLinearFit()

    @with_model(8)
    def testLinearFitConcat(self):
        self._testLinearFit()

    def _testLinearFit(self):
        self.fitmodel.linear = True
        expected = self.fitmodel.linear_parameters.copy()
        self.modify_random(only_linear=True)

        result = self.fitmodel.fit()
        self.assert_result(result, expected)
        self.assertTrue(
            not numpy.allclose(self.fitmodel.ydata, self.fitmodel.yfullmodel)
        )
        self.assertTrue(not numpy.allclose(self.fitmodel.linear_parameters, expected))

        self.fitmodel.use_fit_result(result)
        numpy.testing.assert_allclose(self.fitmodel.ydata, self.fitmodel.yfullmodel)
        numpy.testing.assert_allclose(self.fitmodel.linear_parameters, expected)

    @with_model(1)
    def testNonLinearFit(self):
        self._testNonLinearFit()

    @with_model(8)
    def testNonLinearFitConcat(self):
        self._testNonLinearFit()

    def _testNonLinearFit(self):
        self.fitmodel.linear = False
        expected1 = self.fitmodel.parameters.copy()
        expected2 = self.fitmodel.linear_parameters.copy()
        self.modify_random(only_linear=False)

        # from PyMca5.PyMcaMisc.ProfilingUtils import profile
        # filename = "testNonLinearFit{}.pyprof".format(self.nmodels)
        # with profile(memory=False, filename=filename):
        result = self.fitmodel.fit(full_output=True)

        # TODO: non-linear parameters not precise
        # self.assert_result(result, expected1)
        self.assertTrue(
            not numpy.allclose(self.fitmodel.ydata, self.fitmodel.yfullmodel)
        )
        self.assertTrue(not numpy.allclose(self.fitmodel.parameters, expected1))
        self.assertTrue(not numpy.allclose(self.fitmodel.linear_parameters, expected2))

        self.fitmodel.use_fit_result(result)
        # self.plot()
        self.assert_ymodel()
        # TODO: non-linear parameters not precise
        # numpy.testing.assert_allclose(self.fitmodel.parameters, expected1)
        numpy.testing.assert_allclose(
            self.fitmodel.linear_parameters, expected2, rtol=1e-3
        )

    def assert_result(self, result, expected):
        p = numpy.asarray(result["parameters"])
        pstd = numpy.asarray(result["uncertainties"])
        ll = p - 3 * pstd
        ul = p + 3 * pstd
        self.assertTrue(all((expected >= ll) & (expected <= ul)))

    def assert_ymodel(self):
        a = self.fitmodel.ydata
        b = self.fitmodel.yfullmodel
        mask = (a > 1) & (b > 1)
        self.assertTrue(mask.any())
        numpy.testing.assert_allclose(a[mask], b[mask], rtol=1e-3)

    @with_model(8)
    def testParameterIndex(self):
        # Test parameter index conversion from concatenated model to single model
        nmodels = self.fitmodel.nmodels
        npeaks = self.npeaks
        for linear in [False, True]:
            self.fitmodel.linear = linear
            if linear:
                nshapeparams = 0
            else:
                nshapeparams = self.nshapeparams
            imodels = []
            iparams = []
            for param_idx, param_name in enumerate(self.fitmodel.parameter_names):
                lst = list(
                    self.fitmodel._parameter_model_index(param_idx, linear_only=linear)
                )
                # imodel: model indices
                # iparam: index of the parameter in the corresponing models
                if lst:
                    imodel, iparam = list(zip(*lst))
                else:
                    imodel, iparam = tuple(), tuple()
                if "concentrations" in param_name:
                    offset = 0
                    # Shared parameters
                    self.assertEqual(imodel, tuple(range(nmodels)))
                    self.assertEqual(iparam, (param_idx - offset,) * nmodels)
                else:
                    # Non-shared peak shape parameters
                    offset = npeaks
                    imodels.extend(imodel)
                    iparams.extend(i - offset for i in iparam)
            self.assertEqual(len(imodels), nshapeparams * nmodels)
            expected = numpy.repeat(list(range(nmodels)), nshapeparams)
            self.assertEqual(imodels, expected.tolist())
            expected = numpy.tile(list(range(nshapeparams)), nmodels)
            self.assertEqual(iparams, expected.tolist())

    @with_model(8)
    def testChannelIndex(self):
        # Test model index in concatenated
        strides = [2, 3, 100, 1000, 1100, 1200, 3000]
        for stride in strides:
            x = self.fitmodel.xdata
            x2 = x[::stride]
            access_cnt = numpy.zeros(len(x2), dtype=int)
            vstride = stride
            if stride < 1000:
                vstride = None
            for idx in self.fitmodel._generate_model_data_slices(
                len(x2), stride=vstride
            ):
                chunk = x2[idx]
                access_cnt[idx] += 1
                self.assertTrue(all(numpy.diff(chunk) == stride))
            self.assertTrue(all(access_cnt == 1))


def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testFitModel))
    else:
        # use a predefined order
        testSuite.addTest(testFitModel("testParameterIndex"))
        testSuite.addTest(testFitModel("testChannelIndex"))
        testSuite.addTest(testFitModel("testLinearFit"))
        testSuite.addTest(testFitModel("testNonLinearFit"))
        testSuite.addTest(testFitModel("testLinearFitConcat"))
        testSuite.addTest(testFitModel("testNonLinearFitConcat"))
    return testSuite


def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))


if __name__ == "__main__":
    test()

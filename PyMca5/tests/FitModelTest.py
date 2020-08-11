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
            try:
                return method(self, *args, **kw)
            finally:
                self.validate_model()

        return inner2

    return inner1


class testFitModel(unittest.TestCase):
    def setUp(self):
        self.random_state = numpy.random.RandomState(seed=0)

    def create_model(self, nmodels):
        if nmodels == 1:
            self.fitmodel = SimpleModel.SimpleModel()
        else:
            self.fitmodel = SimpleModel.SimpleConcatModel(ndetectors=nmodels)
        assert not self.fitmodel.linear
        self.init_random()
        self.fitmodel.ydata = self.fitmodel.ymodel
        numpy.testing.assert_array_equal(self.fitmodel.ydata, self.fitmodel.ymodel)
        self.validate_model()

    def init_random(self, **kw):
        if isinstance(self.fitmodel, SimpleModel.SimpleConcatModel):
            for model in self.fitmodel._models:
                self._init_random(model, **kw)
            self.fitmodel.shared_attributes = self.fitmodel.shared_attributes
        else:
            self._init_random(self.fitmodel, **kw)

    def _init_random(self, model, npeaks=10, nchannels=2048, border=0.1):
        """Peaks close to the border will cause the nlls to fail"""
        self.nnonglobals = 4  # zero, gain, wzero, wgain
        self.nglobals = npeaks  # concentrations
        model.xdata_raw = numpy.arange(nchannels)
        model.ydata_raw = numpy.full(nchannels, numpy.nan)
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
        if isinstance(self.fitmodel, SimpleModel.SimpleConcatModel):
            self._modify_random_concat(only_linear=only_linear)
        else:
            self._modify_random(only_linear=only_linear)
        self.validate_model()
        # self.plot()

    def _modify_random(self, only_linear=False):
        if only_linear:
            p = self.fitmodel.parameters
            plin = self.fitmodel.linear_parameters
            plin *= numpy.random.uniform(0.5, 0.8, len(plin))
        else:
            p = self.fitmodel.parameters
            plin = self.fitmodel.linear_parameters
            p *= numpy.random.uniform(0.95, 1, len(p))
            plin *= numpy.random.uniform(0.5, 0.8, len(plin))
            self.fitmodel.parameters = p
            assert numpy.array_equal(self.fitmodel.parameters, p)
        self.fitmodel.linear_parameters = plin
        assert numpy.array_equal(self.fitmodel.linear_parameters, plin)
        return p

    def _modify_random_concat(self, only_linear=False):
        p = self._modify_random(only_linear=only_linear)
        assert not numpy.array_equal(
            self.fitmodel.parameters[: self.nglobals], p[: self.nglobals]
        )
        assert numpy.array_equal(
            self.fitmodel.parameters[self.nglobals :], p[self.nglobals :]
        )

    def validate_model(self):
        self._validate_model(self.fitmodel)
        if isinstance(self.fitmodel, SimpleModel.SimpleConcatModel):
            for model in self.fitmodel._models:
                self._validate_model(model)

    def _validate_model(self, model):
        is_concat = isinstance(model, SimpleModel.SimpleConcatModel)

        assert not model.excluded_parameters
        assert not model.included_parameters
        assert model.nchannels == len(model.xdata)
        assert model.nparameters == len(model.parameters)
        assert model.nlinear_parameters == len(model.linear_parameters)
        arr1 = model.evaluate()
        arr2 = model.evaluate_linear()
        arr3 = sum(model.linear_decomposition())
        arr4 = model.ymodel
        numpy.testing.assert_allclose(arr1, arr2)
        numpy.testing.assert_allclose(arr1, arr3)
        numpy.testing.assert_allclose(arr1, arr4)

        nonlin_names = ["zero", "gain", "wzero", "wgain"]
        lin_names = ["concentrations" + str(i) for i in range(self.nglobals)]
        names = nonlin_names + lin_names
        if is_concat:
            model.validate_shared_attributes()
            assert model.nshared_parameters == self.nglobals
            assert model.nshared_linear_parameters == self.nglobals
            nmodels = model.nmodels
            nonglobal_names = [
                name + str(i) for i in range(nmodels) for name in nonlin_names
            ]
            global_names = lin_names
            names = global_names + nonglobal_names
            n = self.nglobals + self.nnonglobals * nmodels
            assert model.nparameters == n
            assert model.nlinear_parameters == self.nglobals
            assert model.parameter_names == names
            assert model.linear_parameter_names == lin_names
        else:
            assert model.nparameters == self.nglobals + self.nnonglobals
            assert model.nlinear_parameters == self.nglobals
            assert model.parameter_names == names
            assert model.linear_parameter_names == lin_names

    def plot(self):
        import matplotlib.pyplot as plt

        m = self.fitmodel
        derivatives = m.derivatives()
        names = m.parameter_names
        plt.figure()
        plt.plot(m.ydata, label="data")
        plt.plot(m.ymodel, label="model")
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
        assert not numpy.allclose(self.fitmodel.ydata, self.fitmodel.ymodel)
        assert not numpy.allclose(self.fitmodel.linear_parameters, expected)

        self.fitmodel.use_fit_result(result)
        numpy.testing.assert_allclose(self.fitmodel.ydata, self.fitmodel.ymodel)
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
        # with profile(memory=False, filename="testNonLinearFit.pyprof"):
        result = self.fitmodel.fit(full_output=True)

        # TODO: non-linear parameters not precise
        # self.assert_result(result, expected1)
        assert not numpy.allclose(self.fitmodel.ydata, self.fitmodel.ymodel)
        assert not numpy.allclose(self.fitmodel.parameters, expected1)
        assert not numpy.allclose(self.fitmodel.linear_parameters, expected2)

        self.fitmodel.use_fit_result(result)
        # self.plot()
        self.assert_ymodel()
        # TODO: non-linear parameters not precise
        # numpy.testing.assert_allclose(self.fitmodel.parameters, expected1)
        numpy.testing.assert_allclose(
            self.fitmodel.linear_parameters, expected2, rtol=1e-6
        )

    def assert_result(self, result, expected):
        p = numpy.asarray(result["parameters"])
        pstd = numpy.asarray(result["uncertainties"])
        ll = p - 3 * pstd
        ul = p + 3 * pstd
        assert all((expected >= ll) & (expected <= ul))

    def assert_ymodel(self):
        a = self.fitmodel.ydata
        b = self.fitmodel.ymodel
        mask = (a > 1) & (b > 1)
        assert mask.any()
        numpy.testing.assert_allclose(a[mask], b[mask], rtol=1e-3)

    @with_model(8)
    def testParameterIndex(self):
        # Test parameter index conversion from concatenated model to single model
        nmodels = self.fitmodel.nmodels
        nglobals = self.nglobals
        for linear in [False, True]:
            self.fitmodel.linear = linear
            if linear:
                nnonglobals = 0
            else:
                nnonglobals = self.nnonglobals
            imodels = []
            iparams = []
            for param_idx in range(self.fitmodel.nparameters):
                lst = list(
                    self.fitmodel._parameter_model_index(param_idx, linear_only=linear)
                )
                if lst:
                    imodel, iparam = list(zip(*lst))
                else:
                    imodel, iparam = tuple(), tuple()
                if param_idx < nglobals:
                    assert imodel == tuple(range(nmodels))
                    assert iparam == tuple([nnonglobals + param_idx] * nmodels)
                else:
                    imodels.extend(imodel)
                    iparams.extend(iparam)
            assert len(imodels) == nnonglobals * nmodels
            expected = numpy.repeat(list(range(nmodels)), nnonglobals)
            assert imodels == expected.tolist()
            expected = numpy.tile(list(range(nnonglobals)), nmodels)
            assert iparams == expected.tolist()

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
            for idx in self.fitmodel._generate_idx_channels(len(x2), stride=vstride):
                chunk = x2[idx]
                access_cnt[idx] += 1
                assert all(numpy.diff(chunk) == stride)
            assert all(access_cnt == 1)


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

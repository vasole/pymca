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
from contextlib import contextmanager
import numpy
from PyMca5.tests.SimpleModel import SimpleModel
from PyMca5.tests.SimpleModel import SimpleCombinedModel


class testFitModel(unittest.TestCase):
    def setUp(self):
        self.random_state = numpy.random.RandomState(seed=100)

    def testLinearFit(self):
        with self._fit_model_subtests():
            self._test_fit(True)

    def testNonLinearFit(self):
        with self._fit_model_subtests():
            self._test_fit(False)

    def _test_fit(self, linear):
        self.fitmodel.linear = linear
        refined_params = self.fitmodel.get_parameter_values(only_linear=False).copy()
        lin_refined_params = self.fitmodel.get_parameter_values(only_linear=True).copy()
        self.modify_random(only_linear=linear)

        before = self.fitmodel.get_parameter_values(only_linear=False)
        lin_before = self.fitmodel.get_parameter_values(only_linear=True)

        result = self.fitmodel.fit(full_output=True)

        after = self.fitmodel.get_parameter_values(only_linear=False)
        lin_after = self.fitmodel.get_parameter_values(only_linear=True)
        numpy.testing.assert_array_equal(before, after)
        numpy.testing.assert_array_equal(lin_before, lin_after)

        self._assert_model_not_refined(refined_params, lin_refined_params)
        self.fitmodel.use_fit_result(result)
        self._assert_model_refined(refined_params, lin_refined_params)

    def _assert_model_not_refined(self, refined_params, lin_refined_params):
        self.assertTrue(
            not numpy.allclose(self.fitmodel.ydata, self.fitmodel.yfullmodel)
        )
        parameters = self.fitmodel.get_parameter_values(only_linear=False)
        self.assertTrue(not numpy.allclose(parameters, refined_params))
        parameters = self.fitmodel.get_parameter_values(only_linear=True)
        self.assertTrue(not numpy.allclose(parameters, lin_refined_params))

    def _assert_model_refined(self, refined_params, lin_refined_params):
        numpy.testing.assert_allclose(self.fitmodel.ydata, self.fitmodel.yfullmodel)
        parameters = self.fitmodel.get_parameter_values(only_linear=False)
        numpy.testing.assert_allclose(parameters, refined_params)
        parameters = self.fitmodel.get_parameter_values(only_linear=True)
        numpy.testing.assert_allclose(parameters, lin_refined_params)

    def _assert_fit_result(self, result, expected):
        p = numpy.asarray(result["parameters"])
        pstd = numpy.asarray(result["uncertainties"])
        ll = p - 3 * pstd
        ul = p + 3 * pstd
        self.assertTrue(all((expected >= ll) & (expected <= ul)))

    @contextmanager
    def _fit_model_subtests(self):
        for nmodels in (1,):
            with self.subTest(nmodels=nmodels):
                self._create_model(nmodels=nmodels)
                self.validate_model()
                yield
                self.validate_model()

    def _create_model(self, nmodels):
        self.nmodels = nmodels
        self.is_combined_model = nmodels != 1
        if nmodels == 1:
            self.fitmodel = SimpleModel()
        else:
            self.fitmodel = SimpleCombinedModel(ndetectors=nmodels)
        self.assertTrue(not self.fitmodel.linear)

        self.init_random()

        ydata = self.fitmodel.yfullmodel.copy()
        self.fitmodel.ydata = ydata
        numpy.testing.assert_array_equal(self.fitmodel.ydata, ydata)
        numpy.testing.assert_array_equal(self.fitmodel.yfullmodel, ydata)
        numpy.testing.assert_allclose(
            self.fitmodel.yfitmodel, ydata - self.background, atol=1e-12
        )

    def init_random(self, **kw):
        self.npeaks = 13  # concentrations
        self.nshapeparams = 4  # zero, gain, wzero, wgain
        self.nchannels = 2048
        self.border = 0.1  # peak positions not within this border fraction
        self.background = 10
        if self.is_combined_model:
            for model in self.fitmodel.models:
                self._init_random(model)
        else:
            self._init_random(self.fitmodel)

    def _init_random(self, model):
        nchannels = self.nchannels
        model.xdata_raw = numpy.arange(nchannels)
        model.ydata_raw = numpy.full(nchannels, numpy.nan)
        model.ybkg = self.background
        model.xmin = self.random_state.randint(low=0, high=10)
        model.xmax = self.random_state.randint(low=nchannels - 10, high=nchannels)
        model.zero = self.random_state.uniform(low=1, high=1.5)
        model.gain = self.random_state.uniform(low=10e-3, high=11e-3)

        # Peaks too close to the border will cause numerical checking to fail
        a = model.zero
        b = model.zero + model.gain * nchannels
        border = self.border * (b - a)
        a += border
        b -= border
        npeaks = self.npeaks
        model.positions = numpy.linspace(a, b, npeaks)

        model.wzero = self.random_state.uniform(low=0.0, high=0.01)
        model.wgain = self.random_state.uniform(low=0.05, high=0.1)
        model.concentrations = self.random_state.uniform(low=0.5, high=1, size=npeaks)
        model.efficiency = self.random_state.uniform(low=5000, high=6000, size=npeaks)

    def modify_random(self, only_linear=False):
        self._modify_random(only_linear=only_linear)
        self.validate_model()

    def _modify_random(self, only_linear=False):
        porg = self.fitmodel.get_parameter_values(only_linear=False).copy()
        plinorg = self.fitmodel.get_parameter_values(only_linear=True).copy()

        if not only_linear:
            p = porg.copy()
            p *= self.random_state.uniform(0.95, 1, len(p))
            self.fitmodel.set_parameter_values(p, only_linear=False)
            parameters = self.fitmodel.get_parameter_values(only_linear=False)
            numpy.testing.assert_array_equal(parameters, p)

        plin = plinorg.copy()
        plin *= self.random_state.uniform(0.5, 0.8, len(plin))
        self.fitmodel.set_parameter_values(plin, only_linear=True)
        parameters = self.fitmodel.get_parameter_values(only_linear=True)
        numpy.testing.assert_array_equal(parameters, plin)

        if only_linear:
            p = self.fitmodel.get_parameter_values(only_linear=False)

        for group in self.fitmodel.get_parameter_groups(only_linear=False):
            current = p[group.index]
            expected = porg[group.index]
            if only_linear and not group.linear:
                if group.count == 1:
                    self.assertEqual(current, expected, msg=group.name)
                else:
                    self.assertTrue(all(current == expected), msg=group.name)
            else:
                if group.count == 1:
                    self.assertNotEqual(current, expected, msg=group.name)
                else:
                    self.assertFalse(all(current == expected), msg=group.name)

        for group in self.fitmodel.get_parameter_groups(only_linear=True):
            current = plin[group.index]
            expected = plinorg[group.index]
            if group.count == 1:
                self.assertNotEqual(current, expected, msg=group.name)
            else:
                self.assertFalse(all(current == expected), msg=group.name)

        return p

    def validate_model(self):
        self._validate_model(self.fitmodel, self.is_combined_model)
        if self.is_combined_model:
            for model in self.fitmodel.models:
                self._validate_model(model, False)
        self._validate_model(self.fitmodel, self.is_combined_model)

    def _validate_model(self, model, is_combined_model):
        keep_parameters = model.get_parameter_values(only_linear=False).copy()
        keep_linear_parameters = model.get_parameter_values(only_linear=True).copy()

        if not is_combined_model:
            names = model.get_parameter_group_names(only_linear=False)
            expected = {"concentrations", "zero", "gain", "wzero", "wgain"}
            self.assertEqual(set(names), expected)
            names = model.get_parameter_group_names(only_linear=True)
            expected = {"concentrations"}
            self.assertEqual(set(names), expected)

        n = model.ndata
        nexpected = len(model.xdata)
        self.assertEqual(n, nexpected)

        n = model.get_n_parameters(only_linear=False)
        nexpected = len(model.get_parameter_values(only_linear=False))
        self.assertEqual(n, nexpected)

        n = model.get_n_parameters(only_linear=True)
        nexpected = len(model.get_parameter_values(only_linear=True))
        self.assertEqual(n, nexpected)

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

        nonlin_names = {"gain", "wgain", "wzero", "zero"}
        lin_names = {"concentrations" + str(i) for i in range(self.npeaks)}
        names = lin_names | nonlin_names
        if is_combined_model:
            nmodels = model.nmodels
            names = lin_names + [
                name + str(i) for i in range(nmodels) for name in nonlin_names
            ]
            nexpected = self.npeaks + self.nshapeparams * nmodels
        else:
            nexpected = self.npeaks + self.nshapeparams

        n = model.get_n_parameters(only_linear=False)
        self.assertEqual(n, nexpected)
        n = model.get_n_parameters(only_linear=True)
        self.assertEqual(n, self.npeaks)
        mnames = model.get_parameter_names(only_linear=False)
        self.assertEqual(set(mnames), names)
        mnames = model.get_parameter_names(only_linear=True)
        self.assertEqual(set(mnames), lin_names)

        if not is_combined_model:
            for linear in [not model.linear, model.linear]:
                model.linear = linear
                for param_name, calc, numerical in model.compare_derivatives():
                    err_msg = "[only_linear={}] Analytical and numerical derivative of {} are not equal".format(
                        linear, repr(param_name)
                    )
                    numpy.testing.assert_allclose(
                        calc, numerical, err_msg=err_msg, rtol=1e-4
                    )

        parameters = model.get_parameter_values(only_linear=False)
        numpy.testing.assert_array_equal(keep_parameters, parameters)
        parameters = model.get_parameter_values(only_linear=True)
        numpy.testing.assert_array_equal(keep_linear_parameters, parameters)

    def _vis_compare(self, a, b):
        import matplotlib.pyplot as plt

        plt.plot(a)
        plt.plot(b)
        plt.show()

    @contextmanager
    def _profile(self, name):
        from PyMca5.PyMcaMisc.ProfilingUtils import profile

        filename = f"{name}.pyprof"
        with profile(memory=False, filename=filename):
            yield

import unittest
from contextlib import contextmanager
import numpy
from PyMca5.tests.SimpleModel import SimpleModel
from PyMca5.tests.SimpleModel import SimpleCombinedModel
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterType


class testFitModel(unittest.TestCase):
    def setUp(self):
        self.random_state = numpy.random.RandomState(seed=100)

    def testLinearFit(self):
        for _ in self._fit_model_subtests():
            self._test_fit(ParameterType.independent_linear)

    def testNonLinearFit(self):
        for _ in self._fit_model_subtests():
            self._test_fit(None)

    def _test_fit(self, parameter_type):
        self.fitmodel.parameter_type = parameter_type
        refined_params = self.fitmodel.get_parameter_values(parameter_type=None).copy()
        lin_refined_params = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        ).copy()
        self._modify_random(parameter_type=parameter_type)

        before = self.fitmodel.get_parameter_values(parameter_type=None)
        lin_before = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        )

        # with self._profile("test"):
        result = self.fitmodel.fit(full_output=True)

        # Verify the expected fit parameters
        rtol = 1e-3
        if result["parameter_type"]:
            numpy.testing.assert_allclose(
                result["parameters"], lin_refined_params, rtol=rtol
            )
        else:
            numpy.testing.assert_allclose(
                result["parameters"], refined_params, rtol=rtol
            )

        # Check that the model has not been affected
        after = self.fitmodel.get_parameter_values(parameter_type=None)
        lin_after = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        )
        numpy.testing.assert_array_equal(before, after)
        numpy.testing.assert_array_equal(lin_before, lin_after)

        # Modify the fit model
        self._assert_model_not_refined(refined_params, lin_refined_params)
        self.fitmodel.use_fit_result(result)
        self._assert_model_refined(refined_params, lin_refined_params, rtol=rtol)

    def _assert_model_not_refined(self, refined_params, lin_refined_params):
        self.assertTrue(
            not numpy.allclose(self.fitmodel.ydata, self.fitmodel.yfullmodel)
        )
        parameters = self.fitmodel.get_parameter_values(parameter_type=None)
        self.assertTrue(not numpy.allclose(parameters, refined_params))
        parameters = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        )
        self.assertTrue(not numpy.allclose(parameters, lin_refined_params))

    def _assert_model_refined(self, refined_params, lin_refined_params, rtol=1e-7):
        parameters = self.fitmodel.get_parameter_values(parameter_type=None)
        numpy.testing.assert_allclose(parameters, refined_params, rtol=rtol)
        parameters = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        )
        numpy.testing.assert_allclose(parameters, lin_refined_params, rtol=rtol)
        numpy.testing.assert_allclose(self.fitmodel.ydata, self.fitmodel.yfullmodel)

    def _assert_fit_result(self, result, expected):
        p = numpy.asarray(result["parameters"])
        pstd = numpy.asarray(result["uncertainties"])
        ll = p - 3 * pstd
        ul = p + 3 * pstd
        self.assertTrue(all((expected >= ll) & (expected <= ul)))

    def _fit_model_subtests(self):
        for nmodels in [1, 4]:
            with self.subTest(nmodels=nmodels):
                self._create_model(nmodels=nmodels)
                self._validate_models()
                yield
                self._validate_models()

    def _create_model(self, nmodels):
        self.nmodels = nmodels
        self.is_combined_model = nmodels != 1
        if nmodels == 1:
            self.fitmodel = SimpleModel()
        else:
            self.fitmodel = SimpleCombinedModel(ndetectors=nmodels)
        self.assertTrue(not self.fitmodel.parameter_type)

        self._init_random()

        ydata = self.fitmodel.yfullmodel.copy()
        self.fitmodel.ydata = ydata
        numpy.testing.assert_array_equal(self.fitmodel.ydata, ydata)
        numpy.testing.assert_array_equal(self.fitmodel.yfullmodel, ydata)
        numpy.testing.assert_allclose(
            self.fitmodel.yfitmodel, ydata - self.background, atol=1e-12
        )

    def _init_random(self, **kw):
        self.npeaks = 13  # concentrations
        self.nshapeparams = 4  # zero, gain, wzero, wgain
        self.nchannels = 2048
        self.border = 0.1  # peak positions not within this border fraction
        self.background = 10
        if self.is_combined_model:
            for model in self.fitmodel.models:
                self._init_random_model(model)
        else:
            self._init_random_model(self.fitmodel)

    def _init_random_model(self, model):
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

    def _modify_random(self, parameter_type):
        self._modify_random_model(parameter_type)
        self._validate_models()

    def _modify_random_model(self, parameter_type):
        self.assertIn(parameter_type, (None, ParameterType.independent_linear))
        pnonlinorg = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.non_linear
        )
        plinorg = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        )
        pallorg = self.fitmodel.get_parameter_values(parameter_type=None)

        if parameter_type is None:
            pmod = pnonlinorg.copy()
            pmod *= self.random_state.uniform(0.95, 1, len(pmod))
            self.fitmodel.set_parameter_values(
                pmod, parameter_type=ParameterType.non_linear
            )
            parameters = self.fitmodel.get_parameter_values(
                parameter_type=ParameterType.non_linear
            )
            numpy.testing.assert_array_equal(parameters, pmod)

        pmod = plinorg.copy()
        pmod *= self.random_state.uniform(0.5, 0.8, len(pmod))
        self.fitmodel.set_parameter_values(
            pmod, parameter_type=ParameterType.independent_linear
        )
        parameters = self.fitmodel.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        )
        numpy.testing.assert_array_equal(parameters, pmod)

        pall = self.fitmodel.get_parameter_values(parameter_type=None)
        for group in self.fitmodel.get_parameter_groups(parameter_type=None):
            current = pall[group.index]
            expected = pallorg[group.index]
            if parameter_type is None or group.is_independent_linear:
                # Values are expected to be modified
                if group.count == 1:
                    self.assertNotEqual(current, expected, msg=group.name)
                else:
                    self.assertFalse(all(current == expected), msg=group.name)
            else:
                # Values are not expected to be modified
                if group.count == 1:
                    self.assertEqual(current, expected, msg=group.name)
                else:
                    self.assertTrue(all(current == expected), msg=group.name)

    def _validate_models(self):
        self._validate_model(self.fitmodel)
        if self.is_combined_model:
            for model_idx, model in enumerate(self.fitmodel.models):
                self._validate_model(model, model_idx)
        self._validate_model(self.fitmodel)

    def _validate_model(self, model, model_idx=None):
        is_combined_model = self.is_combined_model and model_idx is None
        original_parameters = model.get_parameter_values(parameter_type=None).copy()
        original_linear_parameters = model.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        ).copy()

        nonlin_expected = {"gain", "wgain", "wzero", "zero"}
        if self.is_combined_model:
            if is_combined_model:
                nonlin_expected = {
                    f"detector{model_idx}:{name}"
                    for model_idx in range(self.nmodels)
                    for name in nonlin_expected
                }
            else:
                nonlin_expected = {
                    f"detector{model_idx}:{name}" for name in nonlin_expected
                }
        lin_expected = {"concentrations"}
        all_expected = lin_expected | nonlin_expected
        names = model.get_parameter_group_names(parameter_type=None)
        self.assertEqual(set(names), all_expected)
        names = model.get_parameter_group_names(
            parameter_type=ParameterType.independent_linear
        )
        self.assertEqual(set(names), lin_expected)

        lin_expected = {f"concentrations{i}" for i in range(self.npeaks)}
        all_expected = lin_expected | nonlin_expected
        names = model.get_parameter_names(parameter_type=None)
        self.assertEqual(set(names), all_expected)
        names = model.get_parameter_names(
            parameter_type=ParameterType.independent_linear
        )
        self.assertEqual(set(names), lin_expected)

        n = model.ndata
        nexpected = len(model.xdata)
        self.assertEqual(n, nexpected)

        nexpected = len(model.get_parameter_values(parameter_type=None))
        n = model.get_n_parameters(parameter_type=None)
        self.assertEqual(n, nexpected)

        nexpected = len(
            model.get_parameter_values(parameter_type=ParameterType.independent_linear)
        )
        n = model.get_n_parameters(parameter_type=ParameterType.independent_linear)
        self.assertEqual(n, nexpected)

        arr1 = model.evaluate_fullmodel()
        arr2 = model.evaluate_decomposed_fullmodel()
        arr3 = model.yfullmodel
        numpy.testing.assert_allclose(arr1, arr2)
        numpy.testing.assert_allclose(arr1, arr3)

        arr1 = model.evaluate_fitmodel()
        arr2 = model.evaluate_decomposed_fitmodel()
        arr3 = model.yfitmodel
        arr4 = sum(model.linear_decomposition_fitmodel())
        numpy.testing.assert_allclose(arr1, arr2)
        numpy.testing.assert_allclose(arr1, arr3)
        numpy.testing.assert_allclose(arr1, arr4)

        if is_combined_model:
            nmodels = model.nmodels
            nexpected = self.npeaks + self.nshapeparams * nmodels
        else:
            nexpected = self.npeaks + self.nshapeparams
        n = model.get_n_parameters(parameter_type=None)
        self.assertEqual(n, nexpected)
        n = model.get_n_parameters(parameter_type=ParameterType.independent_linear)
        self.assertEqual(n, self.npeaks)

        rtol = 1e-3
        for parameter_type in (ParameterType.independent_linear, None):
            with model.parameter_type_context(parameter_type):
                noncached = list(model.compare_derivatives())
                cached = list(model.compare_derivatives())
                err_fmt = "[parameter_type={}] {{}} and {{}} derivative of '{{}}' (model: {}) are not equal".format(
                    parameter_type, model_idx
                )
                for deriv, deriv_cached in zip(noncached, cached):
                    param_name, calc, numerical = deriv
                    param_name_cached, calc_cached, numerical_cached = deriv_cached
                    err_msg = err_fmt.format("analytical", "numerical", param_name)
                    self.assertEqual(param_name, param_name_cached)
                    numpy.testing.assert_allclose(
                        calc, numerical, err_msg=err_msg, rtol=rtol
                    )
                    err_msg = err_fmt.format(
                        "cached analytical", "numerical", param_name_cached
                    )
                    numpy.testing.assert_allclose(
                        calc_cached, numerical_cached, err_msg=err_msg, rtol=rtol
                    )
                    err_msg = err_fmt.format("cached", "non-cached ", param_name)
                    numpy.testing.assert_allclose(
                        calc, calc_cached, err_msg=err_msg, rtol=rtol
                    )

        parameters = model.get_parameter_values(parameter_type=None)
        numpy.testing.assert_array_equal(original_parameters, parameters)
        parameters = model.get_parameter_values(
            parameter_type=ParameterType.independent_linear
        )
        numpy.testing.assert_array_equal(original_linear_parameters, parameters)

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

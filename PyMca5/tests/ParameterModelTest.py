import unittest
import numpy
from contextlib import contextmanager
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterModel
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterModelManager
from PyMca5.PyMcaMath.fitting.model.ParameterModel import parameter_group
from PyMca5.PyMcaMath.fitting.model.ParameterModel import linear_parameter_group


class Model1(ParameterModel):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    @parameter_group
    def var1_nonlin(self):
        return self._cfg["var1_nonlin"]

    @var1_nonlin.setter
    def var1_nonlin(self, value):
        self._cfg["var1_nonlin"] = value

    @linear_parameter_group
    def var1_lin(self):
        return self._cfg["var1_lin"]

    @var1_lin.setter
    def var1_lin(self, value):
        self._cfg["var1_lin"] = value

    @var1_lin.counter
    def var1_lin(self):
        return 2

    def __getitem__(self, index):
        return self._linked_instance_mapping[index]


class Model2(Model1):
    @parameter_group
    def var2_nonlin(self):
        return self._cfg["var2_nonlin"]

    @var2_nonlin.setter
    def var2_nonlin(self, value):
        self._cfg["var2_nonlin"] = value

    @linear_parameter_group
    def var2_lin(self):
        return self._cfg["var2_lin"]

    @var2_lin.setter
    def var2_lin(self, value):
        self._cfg["var2_lin"] = value


class ConcatModel(ParameterModelManager):
    def __init__(self):
        cfg1a = {"var1_lin": [11, 11], "var1_nonlin": 12}
        cfg1b = {"var1_lin": [21, 21], "var1_nonlin": 22}
        cfg2a = {
            "var1_lin": [31, 31],
            "var1_nonlin": 32,
            "var2_lin": 41,
            "var2_nonlin": 42,
        }
        cfg2b = {
            "var1_lin": [51, 51],
            "var1_nonlin": 12,
            "var2_lin": 61,
            "var2_nonlin": 62,
        }
        models = {
            "model0": Model1(cfg1a),
            "model1": Model1(cfg1b),
            "model2": Model2(cfg2a),
            "model3": Model2(cfg2b),
        }
        super().__init__(models)
        self._enable_property_link("var1_lin", "var1_nonlin", "var2_lin", "var2_nonlin")

    def __getitem__(self, index):
        while index < 0:
            index += len(self._linked_instance_mapping)
        return self._linked_instance_mapping[f"model{index}"]


class testParameterModel(unittest.TestCase):
    def setUp(self):
        self.concat_model = ConcatModel()

    def test_instantiation(self):
        self.assertFalse(self.concat_model.linear)
        self.assertEqual(self.concat_model.nmodels, 4)

    def test_linear_context(self):
        with self.concat_model.linear_context(True):
            self.assertTrue(self.concat_model.linear)
            for model in self.concat_model.models:
                self.assertTrue(model.linear)

        self.assertFalse(self.concat_model.linear)
        for model in self.concat_model.models:
            self.assertFalse(model.linear)

    def test_parameter_group_names(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            names = self.concat_model[0].get_parameter_group_names(**cacheoptions)
            expected = ("var1_lin", "var1_nonlin")
            self.assertEqual(set(names), set(expected))

            names = self.concat_model[-1].get_parameter_group_names(**cacheoptions)
            expected = ("var1_lin", "var1_nonlin", "var2_lin", "var2_nonlin")
            self.assertEqual(set(names), set(expected))

            names = self.concat_model.get_parameter_group_names(**cacheoptions)
            expected = (
                "var1_lin",
                "var1_nonlin",
                "var2_lin",
                "var2_nonlin",
            )
            self.assertEqual(set(names), set(expected))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_group_names(**cacheoptions)
                    expected = (
                        "model0:var1_lin",
                        "model1:var1_lin",
                        "model2:var1_lin",
                        "model3:var1_lin",
                        "var1_nonlin",
                        "var2_lin",
                        "var2_nonlin",
                    )
                    self.assertEqual(set(names), set(expected))

    def test_linear_parameter_group_names(self):
        for cacheoptions in self._parameterize_linear_test():
            names = self.concat_model[0].get_parameter_group_names(**cacheoptions)
            expected = ("var1_lin",)
            self.assertEqual(set(names), set(expected))

            names = self.concat_model[-1].get_parameter_group_names(**cacheoptions)
            expected = ("var1_lin", "var2_lin")
            self.assertEqual(set(names), set(expected))

            names = self.concat_model.get_parameter_group_names(**cacheoptions)
            expected = ("var1_lin", "var2_lin")
            self.assertEqual(set(names), set(expected))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_group_names(**cacheoptions)
                    expected = (
                        "model0:var1_lin",
                        "model1:var1_lin",
                        "model2:var1_lin",
                        "model3:var1_lin",
                        "var2_lin",
                    )
                    self.assertEqual(set(names), set(expected))

    def test_parameter_names(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            names = self.concat_model[0].get_parameter_names(**cacheoptions)
            expected = ("var1_lin0", "var1_lin1", "var1_nonlin")
            self.assertEqual(set(names), set(expected))

            names = self.concat_model[-1].get_parameter_names(**cacheoptions)
            expected = (
                "var1_lin0",
                "var1_lin1",
                "var1_nonlin",
                "var2_lin",
                "var2_nonlin",
            )
            self.assertEqual(set(names), set(expected))

            names = self.concat_model.get_parameter_names(**cacheoptions)
            expected = (
                "var1_lin0",
                "var1_lin1",
                "var1_nonlin",
                "var2_lin",
                "var2_nonlin",
            )
            self.assertEqual(set(names), set(expected))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_names(**cacheoptions)
                    expected = (
                        "model0:var1_lin0",
                        "model0:var1_lin1",
                        "model1:var1_lin0",
                        "model1:var1_lin1",
                        "model2:var1_lin0",
                        "model2:var1_lin1",
                        "model3:var1_lin0",
                        "model3:var1_lin1",
                        "var1_nonlin",
                        "var2_lin",
                        "var2_nonlin",
                    )
                    self.assertEqual(set(names), set(expected))

    def test_linear_parameter_names(self):
        for cacheoptions in self._parameterize_linear_test():
            names = self.concat_model[0].get_parameter_names(**cacheoptions)
            expected = ("var1_lin0", "var1_lin1")
            self.assertEqual(set(names), set(expected))

            names = self.concat_model[-1].get_parameter_names(**cacheoptions)
            expected = ("var1_lin0", "var1_lin1", "var2_lin")
            self.assertEqual(set(names), set(expected))

            names = self.concat_model.get_parameter_names(**cacheoptions)
            expected = ("var1_lin0", "var1_lin1", "var2_lin")
            self.assertEqual(set(names), set(expected))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_names(**cacheoptions)
                    expected = (
                        "model0:var1_lin0",
                        "model0:var1_lin1",
                        "model1:var1_lin0",
                        "model1:var1_lin1",
                        "model2:var1_lin0",
                        "model2:var1_lin1",
                        "model3:var1_lin0",
                        "model3:var1_lin1",
                        "var2_lin",
                    )
                    self.assertEqual(set(names), set(expected))

    def test_n_parameter(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            n = self.concat_model[0].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 3)

            n = self.concat_model[-1].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 5)

            n = self.concat_model.get_n_parameters(**cacheoptions)
            self.assertEqual(n, 5)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    n = self.concat_model.get_n_parameters(**cacheoptions)
                    self.assertEqual(n, 11)

    def test_n_linear_parameter(self):
        for cacheoptions in self._parameterize_linear_test():
            n = self.concat_model[0].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 2)

            n = self.concat_model[-1].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 3)

            n = self.concat_model.get_n_parameters(**cacheoptions)
            self.assertEqual(n, 3)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    n = self.concat_model.get_n_parameters(**cacheoptions)
                    self.assertEqual(n, 9)

    def test_parameter_constraints(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            arr = self.concat_model[0].get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (3, 3))

            arr = self.concat_model[-1].get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (5, 3))

            arr = self.concat_model.get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (5, 3))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    arr = self.concat_model.get_parameter_constraints(**cacheoptions)
                    self.assertEqual(arr.shape, (11, 3))

    def test_linear_parameter_contraints(self):
        for cacheoptions in self._parameterize_linear_test():
            if not self._cached:
                continue
            arr = self.concat_model[0].get_parameter_constraints(**cacheoptions)
            try:
                self.assertEqual(arr.shape, (2, 3))
            except Exception:
                breakpoint()

            arr = self.concat_model[-1].get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (3, 3))

            arr = self.concat_model.get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (3, 3))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    arr = self.concat_model.get_parameter_constraints(**cacheoptions)
                    self.assertEqual(arr.shape, (9, 3))

    def test_get_parameter_values(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            values = self.concat_model[0].get_parameter_values(**cacheoptions)
            self.assertEqual(values.tolist(), [11, 11, 12])

            values = self.concat_model[-1].get_parameter_values(**cacheoptions)
            self.assertEqual(values.tolist(), [11, 11, 12, 41, 42])

            values = self.concat_model.get_parameter_values(**cacheoptions)
            self.assertEqual(values.tolist(), [11, 11, 12, 41, 42])

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    values = self.concat_model.get_parameter_values(**cacheoptions)
                    self.assertEqual(values.tolist(), [12, 41, 42] + [11] * 8)

    def test_parameter_values(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            self._assert_set_get_parameter_values(
                self.concat_model[0], [11, 11, 12], **cacheoptions
            )
            self._assert_set_get_parameter_values(
                self.concat_model[-1], [11, 11, 12, 41, 42], **cacheoptions
            )
            self._assert_set_get_parameter_values(
                self.concat_model, [11, 11, 12, 41, 42], **cacheoptions
            )
            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    self._assert_set_get_parameter_values(
                        self.concat_model, [12, 41, 42] + [11] * 8, **cacheoptions
                    )

    def test_linear_parameter_values(self):
        for cacheoptions in self._parameterize_linear_test():
            self._assert_set_get_parameter_values(
                self.concat_model[0], [11, 11], **cacheoptions
            )
            self._assert_set_get_parameter_values(
                self.concat_model[-1], [11, 11, 41], **cacheoptions
            )
            self._assert_set_get_parameter_values(
                self.concat_model, [11, 11, 41], **cacheoptions
            )
            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    self._assert_set_get_parameter_values(
                        self.concat_model, [41] + [11] * 8, **cacheoptions
                    )

    def _assert_set_get_parameter_values(self, model, expected, **cacheoptions):
        self._assert_get_parameter_values(model, expected, **cacheoptions)
        with self._protect_parameter_values(**cacheoptions):
            expected2 = list(range(len(expected)))
            model.set_parameter_values(numpy.array(expected2), **cacheoptions)
            self._assert_get_parameter_values(model, expected2, **cacheoptions)
        self._assert_get_parameter_values(model, expected, **cacheoptions)

    def _assert_get_parameter_values(self, model, expected, **cacheoptions):
        values = model.get_parameter_values(**cacheoptions)
        self.assertEqual(values.tolist(), expected)

    def _parameterize_linear_test(self):
        yield from self._parameterize_tests([[True, False], [None, True]])

    def _parameterize_nonlinear_test(self):
        yield from self._parameterize_tests([[False, True], [None, False]])

    def _parameterize_tests(self, linear_options):
        for local_linear, global_linear in linear_options:
            for cached in [True, False]:
                with self.subTest(
                    local_linear=local_linear,
                    global_linear=global_linear,
                    cached=cached,
                ):
                    self.concat_model.linear = global_linear
                    cacheoptions = {"only_linear": local_linear}
                    self._cached = cached
                    if cached:
                        with self.concat_model._propertyCachingContext(**cacheoptions):
                            yield cacheoptions
                    else:
                        yield cacheoptions

    @contextmanager
    def _unlink_var1_lin(self):
        if self._cached:
            yield False
        else:
            self.concat_model._disable_property_link("var1_lin")
            try:
                yield True
            finally:
                self.concat_model._enable_property_link("var1_lin")

    @contextmanager
    def _protect_parameter_values(self, **cacheoptions):
        values = self.concat_model.get_parameter_values(**cacheoptions).copy()
        try:
            yield
        finally:
            self.concat_model.set_parameter_values(values, **cacheoptions)

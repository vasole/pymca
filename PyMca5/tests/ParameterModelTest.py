import unittest
import numpy
from contextlib import contextmanager
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterModel
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterModelManager
from PyMca5.PyMcaMath.fitting.model.ParameterModel import nonlinear_parameter_group
from PyMca5.PyMcaMath.fitting.model.ParameterModel import (
    independent_linear_parameter_group,
)
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterType
from PyMca5.PyMcaMath.fitting.model.ParameterModel import AllParameterTypes


class Model1(ParameterModel):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

    @nonlinear_parameter_group
    def var1_nonlin(self):
        return self._cfg["var1_nonlin"]

    @var1_nonlin.setter
    def var1_nonlin(self, value):
        self._cfg["var1_nonlin"] = value

    @independent_linear_parameter_group
    def var1_lin(self):
        return self._cfg["var1_lin"]

    @var1_lin.setter
    def var1_lin(self, value):
        self._cfg["var1_lin"] = value

    @var1_lin.counter
    def var1_lin(self):
        return 2

    @nonlinear_parameter_group
    def var3_nonlin(self):
        return self._cfg["var3_nonlin"]

    @var3_nonlin.setter
    def var3_nonlin(self, value):
        self._cfg["var3_nonlin"] = value

    @independent_linear_parameter_group
    def var3_lin(self):
        return self._cfg["var3_lin"]

    @var3_lin.setter
    def var3_lin(self, value):
        self._cfg["var3_lin"] = value

    def __getitem__(self, index):
        return self._linked_instance_mapping[index]


class Model2(Model1):
    @nonlinear_parameter_group
    def var2_nonlin(self):
        return self._cfg["var2_nonlin"]

    @var2_nonlin.setter
    def var2_nonlin(self, value):
        self._cfg["var2_nonlin"] = value

    @independent_linear_parameter_group
    def var2_lin(self):
        return self._cfg["var2_lin"]

    @var2_lin.setter
    def var2_lin(self, value):
        self._cfg["var2_lin"] = value


class ConcatModel(ParameterModelManager):
    def __init__(self):
        cfg1a = {
            "var1_lin": [11, 11],
            "var1_nonlin": 12,
            "var3_lin": 101,
            "var3_nonlin": 102,
        }
        cfg1b = {
            "var1_lin": [21, 21],
            "var1_nonlin": 22,
            "var3_lin": 201,
            "var3_nonlin": 202,
        }
        cfg2a = {
            "var1_lin": [31, 31],
            "var1_nonlin": 32,
            "var2_lin": 41,
            "var2_nonlin": 42,
            "var3_lin": 301,
            "var3_nonlin": 302,
        }
        cfg2b = {
            "var1_lin": [51, 51],
            "var1_nonlin": 12,
            "var2_lin": 61,
            "var2_nonlin": 62,
            "var3_lin": 401,
            "var3_nonlin": 402,
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
        self.assertEqual(self.concat_model.nmodels, 4)
        self.assertEqual(self.concat_model.parameter_types, AllParameterTypes)
        for model in self.concat_model.models:
            self.assertEqual(model.parameter_types, AllParameterTypes)

    def test_parameter_type(self):
        for parameter_types in ParameterType:
            for group in self.concat_model.get_parameter_groups(
                parameter_types=parameter_types
            ):
                self.assertEqual(group.type, parameter_types)
        types = set(group.type for group in self.concat_model.get_parameter_groups())
        expected = {ParameterType.independent_linear, ParameterType.non_linear}
        self.assertEqual(types, expected)

    def test_parameter_type_context(self):
        with self.concat_model.parameter_types_context(
            ParameterType.independent_linear
        ):
            self.assertEqual(
                self.concat_model.parameter_types, ParameterType.independent_linear
            )
            for model in self.concat_model.models:
                self.assertTrue(model.parameter_types, ParameterType.independent_linear)

        self.assertEqual(self.concat_model.parameter_types, AllParameterTypes)
        for model in self.concat_model.models:
            self.assertEqual(model.parameter_types, AllParameterTypes)

    def test_parameter_group_names(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            names = self.concat_model[0].get_parameter_group_names(**cacheoptions)
            expected = (
                "var1_lin",
                "var1_nonlin",
                "model0:var3_lin",
                "model0:var3_nonlin",
            )
            self.assertEqual(tuple(names), expected)

            names = self.concat_model[-1].get_parameter_group_names(**cacheoptions)
            expected = (
                "var1_lin",
                "var1_nonlin",
                "var2_lin",
                "var2_nonlin",
                "model3:var3_lin",
                "model3:var3_nonlin",
            )
            self.assertEqual(tuple(names), expected)

            names = self.concat_model.get_parameter_group_names(**cacheoptions)
            expected = (
                "var1_lin",
                "var1_nonlin",
                "var2_lin",
                "var2_nonlin",
                "model0:var3_lin",
                "model0:var3_nonlin",
                "model1:var3_lin",
                "model1:var3_nonlin",
                "model2:var3_lin",
                "model2:var3_nonlin",
                "model3:var3_lin",
                "model3:var3_nonlin",
            )
            self.assertEqual(tuple(names), expected)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_group_names(**cacheoptions)
                    expected = (
                        "var1_nonlin",
                        "var2_lin",
                        "var2_nonlin",
                        "model0:var1_lin",
                        "model0:var3_lin",
                        "model0:var3_nonlin",
                        "model1:var1_lin",
                        "model1:var3_lin",
                        "model1:var3_nonlin",
                        "model2:var1_lin",
                        "model2:var3_lin",
                        "model2:var3_nonlin",
                        "model3:var1_lin",
                        "model3:var3_lin",
                        "model3:var3_nonlin",
                    )
                    self.assertEqual(tuple(names), expected)

    def test_independent_linear_parameter_group_names(self):
        for cacheoptions in self._parameterize_linear_test():
            names = self.concat_model[0].get_parameter_group_names(**cacheoptions)
            expected = ("var1_lin", "model0:var3_lin")
            self.assertEqual(tuple(names), expected)

            names = self.concat_model[-1].get_parameter_group_names(**cacheoptions)
            expected = ("var1_lin", "var2_lin", "model3:var3_lin")
            self.assertEqual(tuple(names), expected)

            names = self.concat_model.get_parameter_group_names(**cacheoptions)
            expected = (
                "var1_lin",
                "var2_lin",
                "model0:var3_lin",
                "model1:var3_lin",
                "model2:var3_lin",
                "model3:var3_lin",
            )
            self.assertEqual(tuple(names), expected)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_group_names(**cacheoptions)
                    expected = (
                        "var2_lin",
                        "model0:var1_lin",
                        "model0:var3_lin",
                        "model1:var1_lin",
                        "model1:var3_lin",
                        "model2:var1_lin",
                        "model2:var3_lin",
                        "model3:var1_lin",
                        "model3:var3_lin",
                    )
                    self.assertEqual(tuple(names), expected)

    def test_parameter_names(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            names = self.concat_model[0].get_parameter_names(**cacheoptions)
            expected = (
                "var1_lin0",
                "var1_lin1",
                "var1_nonlin",
                "model0:var3_lin",
                "model0:var3_nonlin",
            )
            self.assertEqual(tuple(names), expected)

            names = self.concat_model[-1].get_parameter_names(**cacheoptions)
            expected = (
                "var1_lin0",
                "var1_lin1",
                "var1_nonlin",
                "var2_lin",
                "var2_nonlin",
                "model3:var3_lin",
                "model3:var3_nonlin",
            )
            self.assertEqual(tuple(names), expected)

            names = self.concat_model.get_parameter_names(**cacheoptions)
            expected = (
                "var1_lin0",
                "var1_lin1",
                "var1_nonlin",
                "var2_lin",
                "var2_nonlin",
                "model0:var3_lin",
                "model0:var3_nonlin",
                "model1:var3_lin",
                "model1:var3_nonlin",
                "model2:var3_lin",
                "model2:var3_nonlin",
                "model3:var3_lin",
                "model3:var3_nonlin",
            )
            self.assertEqual(tuple(names), expected)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_names(**cacheoptions)
                    expected = (
                        "var1_nonlin",
                        "var2_lin",
                        "var2_nonlin",
                        "model0:var1_lin0",
                        "model0:var1_lin1",
                        "model0:var3_lin",
                        "model0:var3_nonlin",
                        "model1:var1_lin0",
                        "model1:var1_lin1",
                        "model1:var3_lin",
                        "model1:var3_nonlin",
                        "model2:var1_lin0",
                        "model2:var1_lin1",
                        "model2:var3_lin",
                        "model2:var3_nonlin",
                        "model3:var1_lin0",
                        "model3:var1_lin1",
                        "model3:var3_lin",
                        "model3:var3_nonlin",
                    )
                    self.assertEqual(tuple(names), expected)

    def test_linear_parameter_names(self):
        for cacheoptions in self._parameterize_linear_test():
            names = self.concat_model[0].get_parameter_names(**cacheoptions)
            expected = ("var1_lin0", "var1_lin1", "model0:var3_lin")
            self.assertEqual(tuple(names), expected)

            names = self.concat_model[-1].get_parameter_names(**cacheoptions)
            expected = ("var1_lin0", "var1_lin1", "var2_lin", "model3:var3_lin")
            self.assertEqual(tuple(names), expected)

            names = self.concat_model.get_parameter_names(**cacheoptions)
            expected = (
                "var1_lin0",
                "var1_lin1",
                "var2_lin",
                "model0:var3_lin",
                "model1:var3_lin",
                "model2:var3_lin",
                "model3:var3_lin",
            )
            self.assertEqual(tuple(names), expected)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    names = self.concat_model.get_parameter_names(**cacheoptions)
                    expected = (
                        "var2_lin",
                        "model0:var1_lin0",
                        "model0:var1_lin1",
                        "model0:var3_lin",
                        "model1:var1_lin0",
                        "model1:var1_lin1",
                        "model1:var3_lin",
                        "model2:var1_lin0",
                        "model2:var1_lin1",
                        "model2:var3_lin",
                        "model3:var1_lin0",
                        "model3:var1_lin1",
                        "model3:var3_lin",
                    )
                    self.assertEqual(tuple(names), expected)

    def test_n_parameter(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            n = self.concat_model[0].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 5)

            n = self.concat_model[-1].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 7)

            n = self.concat_model.get_n_parameters(**cacheoptions)
            self.assertEqual(n, 13)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    n = self.concat_model.get_n_parameters(**cacheoptions)
                    self.assertEqual(n, 19)

    def test_n_linear_parameter(self):
        for cacheoptions in self._parameterize_linear_test():
            n = self.concat_model[0].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 3)

            n = self.concat_model[-1].get_n_parameters(**cacheoptions)
            self.assertEqual(n, 4)

            n = self.concat_model.get_n_parameters(**cacheoptions)
            self.assertEqual(n, 7)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    n = self.concat_model.get_n_parameters(**cacheoptions)
                    self.assertEqual(n, 13)

    def test_parameter_constraints(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            arr = self.concat_model[0].get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (5, 3))

            arr = self.concat_model[-1].get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (7, 3))

            arr = self.concat_model.get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (13, 3))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    arr = self.concat_model.get_parameter_constraints(**cacheoptions)
                    self.assertEqual(arr.shape, (19, 3))

    def test_linear_parameter_contraints(self):
        for cacheoptions in self._parameterize_linear_test():
            if not self._cached:
                continue
            arr = self.concat_model[0].get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (3, 3))

            arr = self.concat_model[-1].get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (4, 3))

            arr = self.concat_model.get_parameter_constraints(**cacheoptions)
            self.assertEqual(arr.shape, (7, 3))

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    arr = self.concat_model.get_parameter_constraints(**cacheoptions)
                    self.assertEqual(arr.shape, (13, 3))

    def test_get_parameter_values(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            values = self.concat_model[0].get_parameter_values(**cacheoptions)
            self.assertEqual(values.tolist(), [11, 11, 12, 101, 102])

            values = self.concat_model[-1].get_parameter_values(**cacheoptions)
            self.assertEqual(values.tolist(), [11, 11, 12, 41, 42, 401, 402])

            values = self.concat_model.get_parameter_values(**cacheoptions)
            expected = [11, 11, 12, 41, 42, 101, 102, 201, 202, 301, 302, 401, 402]
            self.assertEqual(values.tolist(), expected)

            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    values = self.concat_model.get_parameter_values(**cacheoptions)
                    expected = [
                        12,
                        41,
                        42,
                        11,
                        11,
                        101,
                        102,
                        11,
                        11,
                        201,
                        202,
                        11,
                        11,
                        301,
                        302,
                        11,
                        11,
                        401,
                        402,
                    ]
                    self.assertEqual(values.tolist(), expected)

    def test_parameter_values(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            self._assert_set_get_parameter_values(
                self.concat_model[0], [11, 11, 12, 101, 102], **cacheoptions
            )
            self._assert_set_get_parameter_values(
                self.concat_model[-1], [11, 11, 12, 41, 42, 401, 402], **cacheoptions
            )
            expected = [11, 11, 12, 41, 42, 101, 102, 201, 202, 301, 302, 401, 402]
            self._assert_set_get_parameter_values(
                self.concat_model, expected, **cacheoptions
            )
            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    expected = [
                        12,
                        41,
                        42,
                        11,
                        11,
                        101,
                        102,
                        11,
                        11,
                        201,
                        202,
                        11,
                        11,
                        301,
                        302,
                        11,
                        11,
                        401,
                        402,
                    ]
                    self._assert_set_get_parameter_values(
                        self.concat_model, expected, **cacheoptions
                    )

    def test_linear_parameter_values(self):
        for cacheoptions in self._parameterize_linear_test():
            self._assert_set_get_parameter_values(
                self.concat_model[0], [11, 11, 101], **cacheoptions
            )
            self._assert_set_get_parameter_values(
                self.concat_model[-1], [11, 11, 41, 401], **cacheoptions
            )
            self._assert_set_get_parameter_values(
                self.concat_model, [11, 11, 41, 101, 201, 301, 401], **cacheoptions
            )
            with self._unlink_var1_lin() as unlinked:
                if unlinked:
                    expected = [
                        41,
                        11,
                        11,
                        101,
                        11,
                        11,
                        201,
                        11,
                        11,
                        301,
                        11,
                        11,
                        401,
                    ]
                    self._assert_set_get_parameter_values(
                        self.concat_model, expected, **cacheoptions
                    )

    def test_parameter_property_values(self):
        for cacheoptions in self._parameterize_nonlinear_test():
            self._assertValuesEqual(self.concat_model[0].var1_lin, [11, 11])
            self._assertValuesEqual(self.concat_model[0].var1_nonlin, 12)
            self._assertValuesEqual(self.concat_model[0].var3_lin, 101)
            self._assertValuesEqual(self.concat_model[0].var3_nonlin, 102)

            self._assertValuesEqual(self.concat_model[-1].var1_lin, [11, 11])
            self._assertValuesEqual(self.concat_model[-1].var1_nonlin, 12)
            self._assertValuesEqual(self.concat_model[-1].var2_lin, 41)
            self._assertValuesEqual(self.concat_model[-1].var2_nonlin, 42)
            self._assertValuesEqual(self.concat_model[-1].var3_lin, 401)
            self._assertValuesEqual(self.concat_model[-1].var3_nonlin, 402)

    def _assert_set_get_parameter_values(self, model, expected, **cacheoptions):
        self._assert_get_parameter_values(model, expected, **cacheoptions)
        with self._protect_parameter_values(**cacheoptions):
            expected2 = list(range(len(expected)))
            model.set_parameter_values(numpy.array(expected2), **cacheoptions)
            self._assert_get_parameter_values(model, expected2, **cacheoptions)
        self._assert_get_parameter_values(model, expected, **cacheoptions)

    def _assert_get_parameter_values(self, model, expected, **cacheoptions):
        values = model.get_parameter_values(**cacheoptions)
        self._assertValuesEqual(values, expected)

    def _assertValuesEqual(self, values, expected):
        if isinstance(expected, list):
            self.assertEqual(numpy.asarray(values).tolist(), expected)
        else:
            self.assertEqual(values, expected)

    def _parameterize_linear_test(self):
        yield from self._parameterize_tests(
            [
                [None, ParameterType.independent_linear],
                [ParameterType.independent_linear, AllParameterTypes],
            ]
        )

    def _parameterize_nonlinear_test(self):
        yield from self._parameterize_tests(
            [
                [None, AllParameterTypes],
                [AllParameterTypes, ParameterType.independent_linear],
            ]
        )

    def _parameterize_tests(self, types):
        for local_type, global_type in types:
            for cached in [False, True]:
                with self.subTest(
                    local_type=local_type,
                    global_type=global_type,
                    cached=cached,
                ):
                    self.concat_model.parameter_types = global_type
                    cacheoptions = {"parameter_types": local_type}
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

import unittest
from PyMca5.PyMcaMath.fitting.ParameterModel import ParameterModel
from PyMca5.PyMcaMath.fitting.ParameterModel import ParameterModelContainer
from PyMca5.PyMcaMath.fitting.ParameterModel import parameter_group
from PyMca5.PyMcaMath.fitting.ParameterModel import linear_parameter_group


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


class ConcatModel(ParameterModelContainer):
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
        super().__init__([Model1(cfg1a), Model1(cfg1b), Model2(cfg2a), Model2(cfg2b)])
        self._enable_property_link("var1_lin", "var1_nonlin", "var2_lin", "var2_nonlin")


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
        names = self.concat_model.models[0].get_parameter_group_names()
        expected = ("var1_lin", "var1_nonlin")
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.models[-1].get_parameter_group_names()
        expected = ("var1_lin", "var1_nonlin", "var2_lin", "var2_nonlin")
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.get_parameter_group_names()
        expected = (
            "model0:var1_lin",
            "model0:var1_nonlin",
            "model2:var2_lin",
            "model2:var2_nonlin",
        )
        self.assertEqual(set(names), set(expected))

        self.concat_model._disable_property_link("var1_lin")
        names = self.concat_model.get_parameter_group_names()
        expected = (
            "model0:var1_lin",
            "model0:var1_nonlin",
            "model1:var1_lin",
            "model2:var1_lin",
            "model2:var2_lin",
            "model2:var2_nonlin",
            "model3:var1_lin",
        )
        self.assertEqual(set(names), set(expected))

    def test_linear_parameter_group_names(self):
        names = self.concat_model.models[0].get_parameter_group_names(linear=True)
        expected = ("var1_lin",)
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.models[-1].get_parameter_group_names(linear=True)
        expected = ("var1_lin", "var2_lin")
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.get_parameter_group_names(linear=True)
        expected = ("model0:var1_lin", "model2:var2_lin")
        self.assertEqual(set(names), set(expected))

        self.concat_model._disable_property_link("var1_lin")
        names = self.concat_model.get_parameter_group_names(linear=True)
        expected = (
            "model0:var1_lin",
            "model1:var1_lin",
            "model2:var1_lin",
            "model2:var2_lin",
            "model3:var1_lin",
        )
        self.assertEqual(set(names), set(expected))

    def test_parameter_names(self):
        names = self.concat_model.models[0].get_parameter_names()
        expected = ("var1_lin0", "var1_lin1", "var1_nonlin")
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.models[-1].get_parameter_names()
        expected = ("var1_lin0", "var1_lin1", "var1_nonlin", "var2_lin", "var2_nonlin")
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.get_parameter_names()
        expected = (
            "model0:var1_lin0",
            "model0:var1_lin1",
            "model0:var1_nonlin",
            "model2:var2_lin",
            "model2:var2_nonlin",
        )
        self.assertEqual(set(names), set(expected))

        self.concat_model._disable_property_link("var1_lin")
        names = self.concat_model.get_parameter_names()
        expected = (
            "model0:var1_lin0",
            "model0:var1_lin1",
            "model0:var1_nonlin",
            "model1:var1_lin0",
            "model1:var1_lin1",
            "model2:var1_lin0",
            "model2:var1_lin1",
            "model2:var2_lin",
            "model2:var2_nonlin",
            "model3:var1_lin0",
            "model3:var1_lin1",
        )
        self.assertEqual(set(names), set(expected))

    def test_linear_parameter_names(self):
        names = self.concat_model.models[0].get_parameter_names(linear=True)
        expected = ("var1_lin0", "var1_lin1")
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.models[-1].get_parameter_names(linear=True)
        expected = ("var1_lin0", "var1_lin1", "var2_lin")
        self.assertEqual(set(names), set(expected))

        names = self.concat_model.get_parameter_names(linear=True)
        expected = ("model0:var1_lin0", "model0:var1_lin1", "model2:var2_lin")
        self.assertEqual(set(names), set(expected))

        self.concat_model._disable_property_link("var1_lin")
        names = self.concat_model.get_parameter_names(linear=True)
        expected = (
            "model0:var1_lin0",
            "model0:var1_lin1",
            "model1:var1_lin0",
            "model1:var1_lin1",
            "model2:var1_lin0",
            "model2:var1_lin1",
            "model2:var2_lin",
            "model3:var1_lin0",
            "model3:var1_lin1",
        )
        self.assertEqual(set(names), set(expected))

    def test_n_parameter(self):
        n = self.concat_model.models[0].get_n_parameters()
        self.assertEqual(n, 3)

        n = self.concat_model.models[-1].get_n_parameters()
        self.assertEqual(n, 5)

        n = self.concat_model.get_n_parameters()
        self.assertEqual(n, 5)

        self.concat_model._disable_property_link("var1_lin")
        n = self.concat_model.get_n_parameters()
        self.assertEqual(n, 11)

    def test_n_linear_parameter(self):
        n = self.concat_model.models[0].get_n_parameters(linear=True)
        self.assertEqual(n, 2)

        n = self.concat_model.models[-1].get_n_parameters(linear=True)
        self.assertEqual(n, 3)

        n = self.concat_model.get_n_parameters(linear=True)
        self.assertEqual(n, 3)

        self.concat_model._disable_property_link("var1_lin")
        n = self.concat_model.get_n_parameters(linear=True)
        self.assertEqual(n, 9)

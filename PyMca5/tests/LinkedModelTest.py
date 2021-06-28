import unittest
from collections import Counter
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModel
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModelContainer
from PyMca5.PyMcaMath.fitting.LinkedModel import linked_contextmanager
from PyMca5.PyMcaMath.fitting.LinkedModel import linked_property


class ModelBase(LinkedModel):
    def __init__(self):
        super().__init__()
        self.context_counter = 0

    @linked_contextmanager
    def context(self):
        self.context_counter += 1
        yield

    def fit(self):
        with self.context():
            pass


class Model1(ModelBase):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.reset_counters()

    def reset_counters(self):
        self.get_counter = Counter()
        self.set_counter = Counter()

    @linked_property
    def var1(self):
        self.get_counter["var1"] += 1
        return self._cfg.get("var1")

    @var1.setter
    def var1(self, value):
        self.set_counter["var1"] += 1
        self._cfg["var1"] = value


class Model2(Model1):
    @linked_property
    def var2(self):
        self.get_counter["var2"] += 1
        return self._cfg.get("var2")

    @var2.setter
    def var2(self, value):
        self.set_counter["var2"] += 1
        self._var2 = value
        self._cfg["var2"] = value


class ConcatModel(LinkedModelContainer, ModelBase):
    def __init__(self):
        cfg1a = {"var1": 1}
        cfg1b = {"var1": 2}
        cfg2a = {"var1": 3, "var2": 4}
        cfg2b = {"var1": 5, "var2": 6}
        super().__init__([Model1(cfg1a), Model1(cfg1b), Model2(cfg2a), Model2(cfg2b)])

    def reset_counters(self):
        for model in self.linked_instances:
            model.reset_counters()

    def link(self):
        self.enable_property_link("var1", "var2")
        self.reset_counters()

    def unlink(self):
        self.disable_property_link("var1", "var2")
        self.reset_counters()


class testLinkedModel(unittest.TestCase):
    def setUp(self):
        self.concat_model = ConcatModel()

    def test_links(self):
        """establish links"""
        nlinked = len(self.concat_model.linked_instances) - 1
        for model in self.concat_model.linked_instances:
            self.assertEqual(len(model.linked_instances), nlinked)

    def test_init_properties(self):
        """initial property values"""
        self.assert_property_values("var1", 1, [1, 2, 3, 5])
        self.assert_property_values("var2", 4, [4, 6])

    def test_enable_property_link_syncing(self):
        """maximal 1 get/set of a linked property when enabling linking"""
        for i, model in enumerate(self.concat_model.linked_instances):
            model.var1 = 100 + i
            if model.has_linked_property("var2"):
                model.var2 = 200 + i
        self.assert_property_values("var1", 100, [100, 101, 102, 103])
        self.assert_property_values("var2", 202, [202, 203])
        self.concat_model.reset_counters()

        self.concat_model.enable_property_link("var1")
        getmodel = self.concat_model.instance_with_linked_property("var1")
        for model in self.concat_model.linked_instances:
            if model is getmodel:
                self.assertEqual(model.get_counter["var1"], 1)
                self.assertTrue(model.set_counter["var1"] <= 1)
            else:
                self.assertEqual(model.get_counter["var1"], 0)
                self.assertEqual(model.set_counter["var1"], 1)
            self.assertEqual(model.get_counter["var2"], 0)
            self.assertEqual(model.set_counter["var2"], 0)
        self.assert_property_values("var1", 100)
        self.assert_property_values("var2", 202, [202, 203])

    def test_contexts_concat(self):
        """entering a linked context manager of a container"""
        self.concat_model.fit()
        self.assertEqual(self.concat_model.context_counter, 1)
        for model in self.concat_model.linked_instances:
            model.context_counter, 1

    def test_contexts_single(self):
        """entering a linked context manager of a single instance"""
        model = self.concat_model.linked_instances[2]
        model.fit()
        self.assertEqual(model.context_counter, 1)
        for model in self.concat_model.linked_instances:
            model.context_counter, 1

    def test_get_var1_concat(self):
        """getting a linked property (present in all models) from a container"""
        self.concat_model.link()
        self.assertEqual(self.concat_model.get_linked_property_value("var1"), 1)
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(
                model.get_counter["var1"], 1 if i == 0 else 0, msg=f"model{i}"
            )
            self.assertEqual(model.set_counter["var1"], 0, msg=f"model{i}")

    def test_get_var1_single(self):
        """getting a linked property (present in all models) from a single instance"""
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        self.assertEqual(model.var1, 1)
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(
                model.get_counter["var1"], 1 if i == 2 else 0, msg=f"model{i}"
            )
            self.assertEqual(model.set_counter["var1"], 0, msg=f"model{i}")

    def test_get_var2_concat(self):
        """getting a linked property (present in some models) from a container"""
        self.concat_model.link()
        self.assertEqual(self.concat_model.get_linked_property_value("var2"), 4)
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(
                model.get_counter["var2"], 1 if i == 2 else 0, msg=f"model{i}"
            )
            self.assertEqual(model.set_counter["var2"], 0, msg=f"model{i}")

    def test_get_var2_single(self):
        """getting a linked property (present in some models) from a single instance"""
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        self.assertEqual(model.var2, 4)
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(
                model.get_counter["var2"], 1 if i == 2 else 0, msg=f"model{i}"
            )
            self.assertEqual(model.set_counter["var2"], 0, msg=f"model{i}")

    def test_set_var1_concat(self):
        """setting a linked property (present in all models) from a container"""
        self.concat_model.link()
        self.concat_model.set_linked_property_value("var1", 100)
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(model.get_counter["var1"], 0, msg=f"model{i}")
            self.assertEqual(model.set_counter["var1"], 1, msg=f"model{i}")
        self.assert_property_values("var1", 100)

    def test_set_var1_single(self):
        """setting a linked property (present in all models) from a single instance"""
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        model.var1 = 1
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(model.get_counter["var1"], 0, msg=f"model{i}")
            self.assertEqual(model.set_counter["var1"], 1, msg=f"model{i}")

    def test_set_var2_concat(self):
        """setting a linked property (present in some models) from a container"""
        self.concat_model.link()
        self.concat_model.set_linked_property_value("var2", 100)
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(model.get_counter["var2"], 0, msg=f"model{i}")
            self.assertEqual(
                model.set_counter["var2"], 1 if i > 1 else 0, msg=f"model{i}"
            )
        self.assert_property_values("var2", 100)

    def test_set_var2_single(self):
        """setting a linked property (present in some models) from a single instance"""
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        model.var2 = 100
        for i, model in enumerate(self.concat_model.linked_instances):
            self.assertEqual(model.get_counter["var2"], 0, msg=f"model{i}")
            self.assertEqual(
                model.set_counter["var2"], 1 if i > 1 else 0, msg=f"model{i}"
            )

    def assert_property_values(self, name, value, values=None):
        self.assertEqual(
            self.concat_model.get_linked_property_value(name), value, msg=name
        )
        if not isinstance(values, list):
            values = [value] * len(self.concat_model.linked_instances)
        for model, v in zip(
            self.concat_model.instances_with_linked_property(name), values
        ):
            self.assertEqual(getattr(model, name), v, msg=name)

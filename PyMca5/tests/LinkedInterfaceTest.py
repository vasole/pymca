import unittest
from collections import Counter
from PyMca5.PyMcaMath.fitting.LinkedInterface import LinkedInterface
from PyMca5.PyMcaMath.fitting.LinkedInterface import LinkedContainerInterface
from PyMca5.PyMcaMath.fitting.LinkedInterface import linked_contextmanager
from PyMca5.PyMcaMath.fitting.LinkedInterface import linked_property


class ModelBase(LinkedInterface):
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


class ConcatModel(LinkedContainerInterface, ModelBase):
    def __init__(self):
        cfg1a = {"var1": 1}
        cfg1b = {"var1": 2}
        cfg2a = {"var1": 3, "var2": 4}
        cfg2b = {"var1": 5, "var2": 6}
        super().__init__([Model1(cfg1a), Model1(cfg1b),
            Model2(cfg2a), Model2(cfg2b)])

    def reset_counters(self):
        for m in self.linked_instances:
            m.reset_counters()

    def link(self):
        self.enable_property_link("var1", "var2")
        self.reset_counters()

    def unlink(self):
        self.disable_property_link("var1", "var2")
        self.reset_counters()


class testLinkedInterface(unittest.TestCase):
    def setUp(self):
        self.concat_model = ConcatModel()

    def test_links(self):
        """establish links
        """
        nlinked = len(self.concat_model.linked_instances) - 1
        for m in self.concat_model.linked_instances:
            self.assertEqual(len(m.linked_instances),  nlinked)

    def test_init_properties(self):
        """initial property values
        """
        self.assert_property_values("var1", 1, [1, 2, 3, 5])
        self.assert_property_values("var2", 4, [4, 6])

    def test_enable_property_link_syncing(self):
        """maximal 1 get/set of a linked property when enabling linking
        """
        self.concat_model.enable_property_link("var1", "var2")
        for name in ("var1", "var2"):
            for m in self.concat_model.linked_instances:
                self.assertTrue(m.get_counter[name] <= 1)
                self.assertTrue(m.set_counter[name] <= 1)
        self.assert_synced_values()

    def test_contexts_concat(self):
        """entering a linked context manager of a container
        """
        self.concat_model.fit()
        self.assertEqual(self.concat_model.context_counter,  1)
        for m in self.concat_model.linked_instances:
            m.context_counter,  1

    def test_contexts_single(self):
        """entering a linked context manager of a single instance
        """
        model = self.concat_model.linked_instances[2]
        model.fit()
        self.assertEqual(model.context_counter,  1)
        for m in self.concat_model.linked_instances:
            m.context_counter,  1

    def test_get_var1_concat(self):
        """getting a linked property (present in all models) from a container
        """
        self.concat_model.link()
        self.assertEqual(self.concat_model.get_linked_property("var1"),  1)
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var1"], 1 if i == 0 else 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var1"], 0, msg=f"model{i}")

    def test_get_var1_single(self):
        """getting a linked property (present in all models) from a single instance
        """
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        self.assertEqual(model.var1, 1)
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var1"], 1 if i == 2 else 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var1"], 0, msg=f"model{i}")

    def test_get_var2_concat(self):
        """getting a linked property (present in some models) from a container
        """
        self.concat_model.link()
        self.assertEqual(self.concat_model.get_linked_property("var2"),  4)
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var2"], 1 if i == 2 else 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var2"], 0, msg=f"model{i}")

    def test_get_var2_single(self):
        """getting a linked property (present in some models) from a single instance
        """
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        self.assertEqual(model.var2, 4)
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var2"], 1 if i == 2 else 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var2"], 0, msg=f"model{i}")

    def test_set_var1_concat(self):
        """setting a linked property (present in all models) from a container
        """
        self.concat_model.link()
        self.concat_model.set_linked_property("var1", 100)
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var1"], 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var1"], 1, msg=f"model{i}")
        self.assert_synced_values(var1=100)

    def test_set_var1_single(self):
        """setting a linked property (present in all models) from a single instance
        """
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        model.var1 = 1
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var1"], 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var1"], 1, msg=f"model{i}")

    def test_set_var2_concat(self):
        """setting a linked property (present in some models) from a container
        """
        self.concat_model.link()
        self.concat_model.set_linked_property("var2",  100)
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var2"], 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var2"], 1 if i > 1 else 0, msg=f"model{i}")
        self.assert_synced_values(var2=100)

    def test_set_var2_single(self):
        """setting a linked property (present in some models) from a single instance
        """
        self.concat_model.link()
        model = self.concat_model.linked_instances[2]
        model.var2 =  100
        for i, m in enumerate(self.concat_model.linked_instances):
            self.assertEqual(m.get_counter["var2"], 0, msg=f"model{i}")
            self.assertEqual(m.set_counter["var2"], 1 if i > 1 else 0, msg=f"model{i}")

    def assert_synced_values(self, var1=1, var2=4):
        self.assert_property_values("var1", var1)
        self.assert_property_values("var2", var2)

    def assert_property_values(self, name, value, values=None):
        self.assertEqual(self.concat_model.get_linked_property(name), value, msg=name)
        if values is None:
            values = [value] * len(self.concat_model.linked_instances)
        for m, v in zip(self.concat_model.instances_with_linked_property(name), values):
            self.assertEqual(getattr(m, name), v, msg=name)

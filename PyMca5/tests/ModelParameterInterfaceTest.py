import unittest
from collections import Counter
from PyMca5.PyMcaMath.fitting.ModelParameterInterface import ModelParameterInterface
from PyMca5.PyMcaMath.fitting.ModelParameterInterface import (
    ConcatModelParameterInterface,
)
from PyMca5.PyMcaMath.fitting.ModelParameterInterface import parameter_group
from PyMca5.PyMcaMath.fitting.ModelParameterInterface import linear_parameter_group


class Model(ModelParameterInterface):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._shared_param = 2
        self._shared_linear_param = 3
        self._param = 4
        self._linear_param = 5
        self.reset_counters()

    def reset_counters(self):
        self.get_counter = Counter()
        self.set_counter = Counter()

    @parameter_group
    def shared_param(self):
        self.get_counter["shared_param"] += 1
        return self._cfg.get("shared_param", None)

    @shared_param.setter
    def shared_param(self, value):
        self.set_counter["shared_param"] += 1
        self._cfg["shared_param"] = value

    @linear_parameter_group
    def shared_linear_param(self):
        self.get_counter["shared_linear_param"] += 1
        return self._cfg.get("shared_linear_param", None)

    @shared_linear_param.setter
    def shared_linear_param(self, value):
        self.set_counter["shared_linear_param"] += 1
        self._cfg["shared_linear_param"] = value

    @parameter_group
    def param(self):
        self.get_counter["param"] += 1
        return self._cfg.get("param", None)

    @param.setter
    def param(self, value):
        self.set_counter["param"] += 1
        self._cfg["param"] = value

    @linear_parameter_group
    def linear_param(self):
        self.get_counter["linear_param"] += 1
        return self._cfg.get("linear_param", None)

    @linear_param.setter
    def linear_param(self, value):
        self.set_counter["linear_param"] += 1
        self._cfg["linear_param"] = value


class ConcatModel(ConcatModelParameterInterface):
    def __init__(self):
        cfgs = list()
        for i in range(2):
            off = i * 4
            cfg = {
                "shared_param": off + 1,
                "shared_linear_param": off + 2,
                "param": off + 3,
                "linear_param": off + 4,
            }
            cfgs.append(cfg)
        super().__init__([Model(cfg) for cfg in cfgs])
        self._enable_property_link("shared_param", "shared_linear_param")
        self.reset_counters()

    def reset_counters(self):
        for m in self._linked_instances:
            m.reset_counters()


class testModelParameterInterface(unittest.TestCase):
    def setUp(self):
        self.concat_model = ConcatModel()

    def test_parameter_names(self):
        self.assertEqual(self.concat_model.get_parameter_names(), ("shared_param",))

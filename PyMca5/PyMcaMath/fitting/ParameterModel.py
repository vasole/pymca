import typing
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModel
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModelManager
from PyMca5.PyMcaMath.fitting.LinkedModel import linked_property
from PyMca5.PyMcaMath.fitting.CachingModel import CachedPropertiesModel
from PyMca5.PyMcaMath.fitting.CachingModel import cached_property


class parameter_group(cached_property, linked_property):
    """Usage:

    .. highlight:: python
    .. code-block:: python

        class MyClass(Model):

            def __init__(self):
                self._myparam = 0.

            @parameter_group
            def myparam(self):
                return self._myparam

            @myparam.setter  # optional
            def myparam(self, value):
                self._myparam = value

            @myparam.counter  # optional
            def myparam(self):
                return 1

            @myparam.constraints  # optional
            def myparam(self):
                return 1, 0, 0
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.fcount = self._fcount_default()
        self.fconstraints = self._fconstraints_default()

    def counter(self, fcount):
        self.fcount = fcount
        return self

    def constraints(self, fconstraints):
        self.fconstraints = fconstraints
        return self

    def _fcount_default(self):
        def fcount(oself):
            try:
                return len(self.fget(oself))
            except TypeError:
                return 1

        return fcount

    def _fconstraints_default(self):
        def fconstraints(oself):
            return numpy.zeros((self.fcount(oself), 3))

        return fconstraints


class linear_parameter_group(parameter_group):
    pass


@dataclass(frozen=True, eq=True)
class ParameterGroupId:
    name: str
    linear: bool = field(compare=False, hash=False)
    linked: bool = field(compare=False, hash=False)
    count: int = field(compare=False, hash=False)
    start_index: int = field(compare=False, hash=False)
    index: "typing.Any" = field(compare=False, hash=False)
    property_name: str = field(compare=False, hash=False)
    instance_key: "typing.Any" = field(compare=False, hash=False)

    def _iter_parameter_names(self):
        if self.count > 1:
            for i in range(self.count):
                yield self.name + str(i)
        elif self.count == 1:
            yield self.name


class ParameterModelBase(CachedPropertiesModel):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._linear = False

    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, value):
        self._linear = value

    @contextmanager
    def linear_context(self, linear):
        keep = self.linear
        self.linear = linear
        try:
            yield
        finally:
            self.linear = keep

    def _property_cache_key(self, linear=None, **paramtype):
        if linear is None:
            linear = self.linear
        return linear

    def _create_empty_property_values_cache(self, key, **paramtype):
        return numpy.zeros(self.get_n_parameters(**paramtype))

    def _property_cache_index(self, group, **paramtype):
        return group.index

    def get_parameter_names(self, **paramtype):
        return tuple(self._iter_parameter_names(**paramtype))

    def _iter_parameter_names(self, **paramtype):
        for group in self._iter_parameter_groups(**paramtype):
            yield from group._iter_parameter_names()

    def get_n_parameters(self, **paramtype):
        return sum(group.count for group in self._iter_parameter_groups(**paramtype))

    def get_parameter_values(self, **paramtype):
        return self._get_property_values(**paramtype)

    def set_parameter_values(self, values, **paramtype):
        self._set_property_values(values, **paramtype)

    def get_parameter_group_value(self, group, **paramtype):
        return self._get_property_value(group, **paramtype)

    def set_parameter_group_value(self, group, value, **paramtype):
        self._set_property_value(group, value, **paramtype)

    def get_parameter_groups(self, **paramtype):
        return tuple(self._iter_parameter_groups(**paramtype))

    def get_parameter_group_names(self, **paramtype):
        return tuple(group.name for group in self._iter_parameter_groups(**paramtype))

    def _iter_parameter_groups(self, **paramtype):
        """
        :yields ParameterGroupId:
        """
        yield from self._iter_cached_property_names(**paramtype)


class ParameterModel(ParameterModelBase, LinkedModel):
    def _instance_cached_property_names(self, linear=None, linked=None, tracker=None):
        """
        :yields ParameterGroupId:
        """
        count = None
        index = None
        if tracker is None:
            start_index = 0
        else:
            start_index = tracker.start_index
        cls = type(self)
        for property_name in self._cached_property_names():
            prop = getattr(cls, property_name)
            if not isinstance(prop, parameter_group):
                raise TypeError(
                    "Currently only 'parameter_group' properties support caching"
                )

            group_is_linked = prop.propagate
            if linked is not None:
                if group_is_linked is not linked:
                    continue

            group_is_linear = isinstance(prop, linear_parameter_group)
            if linear is None:
                linear = self.linear
            if linear:
                if not group_is_linear:
                    continue

            count = prop.fcount(self)
            if not count:
                continue

            if count > 1:
                index = slice(start_index, start_index + count)
            elif count == 1:
                index = start_index
            else:
                index = None

            instance_key = self._linked_instance_to_key
            if group_is_linked:
                name = property_name
            else:
                name = f"{instance_key}:{property_name}"

            group = ParameterGroupId(
                name=name,
                linear=group_is_linear,
                linked=group_is_linked,
                property_name=property_name,
                instance_key=instance_key,
                count=count,
                start_index=start_index,
                index=index,
            )
            if tracker is None:
                yield group
                start_index += count
            elif tracker.is_new_group(group):
                yield group
                start_index = tracker.start_index

    @linked_property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, value):
        self._linear = value

    def _get_noncached_property_value(self, group):
        return getattr(self, group.property_name)

    def _set_noncached_property_value(self, group, value):
        setattr(self, group.property_name, value)


class Tracker:
    def __init__(self):
        self._start_index = 0
        self._encountered = set()

    @property
    def start_index(self):
        return self._start_index

    def is_new_group(self, group):
        if group in self._encountered:
            return False
        self._encountered.add(group)
        self._start_index += group.count
        return True


class ParameterModelContainer(ParameterModelBase, LinkedModelManager):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._enable_property_link("linear")
        for model in self.models:
            model._cache_manager = self

    @property
    def models(self):
        return self._linked_instances

    @property
    def model_mapping(self):
        return self._linked_instance_mapping

    @property
    def nmodels(self):
        return len(self.model_mapping)

    @property
    def linear(self):
        return self._get_linked_property_value("linear")

    @linear.setter
    def linear(self, value):
        self._set_linked_property_value("linear", value)

    def _instance_cached_property_names(self, **paramtype):
        """
        :yields ParameterGroupId:
        """
        # Shared parameters
        tracker = Tracker()
        start_index = 0
        for model in self.models:
            yield from model._iter_parameter_groups(
                linked=True, tracker=tracker, **paramtype
            )
        # Non-shared parameters
        for model in self.models:
            yield from model._iter_parameter_groups(
                linked=False, tracker=tracker, **paramtype
            )

    def _get_noncached_property_value(self, group):
        instance = self._linked_key_to_instance(group.instance_key)
        return getattr(instance, group.property_name)

    def _set_noncached_property_value(self, group, value):
        instance = self._linked_key_to_instance(group.instance_key)
        setattr(instance, group.property_name, value)

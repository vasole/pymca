from typing import Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy
from enum import Flag

from .CachingLinkedModel import CachedPropertiesLinkModel
from .LinkedModel import LinkedModelManager
from .LinkedModel import linked_property
from .CachingModel import CachedPropertiesModel
from .CachingModel import cached_property


ParameterType = Flag("ParameterType", "non_linear dependent_linear independent_linear")
LinearParameterTypes = ParameterType.dependent_linear | ParameterType.independent_linear
AllParameterTypes = (
    ParameterType.non_linear
    | ParameterType.dependent_linear
    | ParameterType.independent_linear
)


class _parameter_group(cached_property, linked_property):
    """Specify a getter and setter for a group of fit parameters.
    The counter and constraints are optional. When the counter
    returns zero, the variable is excluded from the fit parameters.
    Use CFIXED in the constraints if the parameters should be
    included but not optimized.

    Usage:

    .. highlight:: python
    .. code-block:: python

        class MyClass(Model):

            def __init__(self):
                self._myparam = 0.

            @nonlinear_parameter_group
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

    TYPE = NotImplemented

    def __init__(self, *args, **kw):
        self.fcount = self._fcount_default()
        self.fconstraints = self._fconstraints_default()
        super().__init__(*args, **kw)

    def counter(self, fcount):
        self.fcount = fcount
        return self

    def constraints(self, fconstraints):
        self.fconstraints = fconstraints
        return self

    def _fcount_default(self):
        def fcount(oself):
            values = self.fget(oself, nocache=True)
            try:
                return len(values)
            except TypeError:
                return 1

        return fcount

    def _fconstraints_default(self):
        def fconstraints(oself):
            return numpy.zeros((self.fcount(oself), 3))

        return fconstraints


class nonlinear_parameter_group(_parameter_group):
    TYPE = ParameterType.non_linear


class dependent_linear_parameter_group(_parameter_group):
    TYPE = ParameterType.dependent_linear


class independent_linear_parameter_group(_parameter_group):
    TYPE = ParameterType.independent_linear


@dataclass(frozen=True, eq=True)
class ParameterGroupId:
    name: str
    property_name: str = field(compare=False, hash=False)
    type: ParameterType = field(compare=False, hash=False)
    linked: bool = field(compare=False, hash=False)
    count: int = field(compare=False, hash=False)
    start_index: int = field(compare=False, hash=False)
    stop_index: int = field(compare=False, hash=False)
    index: Any = field(compare=False, hash=False)
    instance_key: Any = field(compare=False, hash=False)
    get_constraints: Any = field(compare=False, hash=False, repr=False)

    def parameter_names(self):
        if self.count > 1:
            for i in range(self.count):
                yield self.name + str(i)
        elif self.count == 1:
            yield self.name

    def contains_parameter_index(self, param_idx):
        return self.start_index <= param_idx < self.stop_index

    def parameter_index_in_group(self, param_idx):
        if self.contains_parameter_index(param_idx):
            return param_idx - self.start_index
        return None

    @property
    def is_linear(self):
        return self.type != ParameterType.non_linear

    @property
    def is_independent_linear(self):
        return self.type == ParameterType.independent_linear


class ParameterModelBase(CachedPropertiesModel):
    """Interface for all models that manage fit parameters"""

    @property
    def parameter_types(self):
        raise NotImplementedError

    @parameter_types.setter
    def parameter_types(self, value):
        raise NotImplementedError

    @property
    def only_linear_parameters(self):
        return not bool(self.parameter_types & ParameterType.non_linear)

    @contextmanager
    def parameter_types_context(self, value=None):
        keep = self.parameter_types
        if value is not None:
            self.parameter_types = value
        try:
            yield
        finally:
            self.parameter_types = keep

    def _property_cache_key(self, parameter_types=None, **paramtype):
        if parameter_types is None:
            parameter_types = self.parameter_types
        elif not isinstance(parameter_types, ParameterType):
            raise TypeError(parameter_types, "must be None or ParameterType")
        return parameter_types

    def _create_empty_property_values_cache(self, key, **paramtype):
        return numpy.zeros(self.get_n_parameters(**paramtype))

    def _property_index_from_id(self, group, **cacheoptions):
        return group.index

    def _property_name_from_id(self, group):
        return group.property_name

    def get_parameter_names(self, **paramtype):
        return tuple(self._iter_parameter_names(**paramtype))

    def _iter_parameter_names(self, **paramtype):
        for group in self._iter_parameter_groups(**paramtype):
            yield from group.parameter_names()

    def get_n_parameters(self, **paramtype):
        return sum(group.count for group in self._iter_parameter_groups(**paramtype))

    def get_parameter_values(self, **paramtype):
        return self._get_property_values(**paramtype)

    def set_parameter_values(self, values, **paramtype):
        self._set_property_values(values, **paramtype)

    def get_parameter_constraints(self, **paramtype):
        """
        :returns array: nparams x 3
        """
        return numpy.vstack(
            tuple(
                self._normalize_constraints(group.get_constraints())
                for group in self._iter_parameter_groups(**paramtype)
            )
        )

    @staticmethod
    def _normalize_constraints(constraints):
        constraints = numpy.atleast_1d(constraints)
        if constraints.ndim not in (1, 2):
            raise ValueError(
                "Parameter group constraints must be of shape (3,) or (nparams, 3)"
            )
        if constraints.shape[-1] != 3:
            raise ValueError(
                "Parameter group constraints must be of shape (3,) or (nparams, 3)"
            )
        return constraints.tolist()

    def get_parameter_group_value(self, group, **paramtype):
        return self._get_property_value(group, **paramtype)

    def set_parameter_group_value(self, group, value, **paramtype):
        self._set_property_value(group, value, **paramtype)

    def get_parameter_groups(self, **paramtype):
        return tuple(self._iter_parameter_groups(**paramtype))

    def _property_id_from_name(self, property_name):
        return self._group_from_parameter_name(property_name)

    def get_parameter_group_names(self, **paramtype):
        return tuple(group.name for group in self._iter_parameter_groups(**paramtype))

    def _iter_parameter_groups(self, **paramtype):
        """This will only yield the groups with count > 0.

        :yields ParameterGroupId:
        """
        yield from self._iter_cached_property_ids(**paramtype)

    def _group_from_parameter_index(self, param_idx, **paramtype):
        for group in self._iter_parameter_groups(**paramtype):
            if group.start_index <= param_idx < group.stop_index:
                return group
        return None

    def _group_from_parameter_name(self, property_name, **paramtype):
        for group in self._iter_parameter_groups(**paramtype):
            if group.property_name == property_name:
                return group
        return None


class ParameterModel(CachedPropertiesLinkModel, ParameterModelBase):
    """Model that implements fit parameters"""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.__parameter_types = AllParameterTypes

    def _iter_cached_property_ids(self, **paramtype):
        instance_key = self._linked_instance_to_key
        for propid in super()._iter_cached_property_ids(**paramtype):
            if propid.linked or propid.instance_key == instance_key:
                yield propid

    def __iter_parameter_group_properties(self):
        cls = type(self)
        for property_name in self._cached_property_names():
            prop = getattr(cls, property_name)
            if not isinstance(prop, _parameter_group):
                raise TypeError(
                    "Currently only parameter _group properties support caching"
                )
            yield property_name, prop

    def _instance_cached_property_ids(
        self, parameter_types=None, linked=None, tracker=None
    ):
        """
        :param parameter_types ParameterType: only these parameter types
        :param linked bool: linked parameters or unlinked parameters
        :param tracker _IterGroupTracker:
        :yields ParameterGroupId:
        """
        if parameter_types is None:
            parameter_types = self.parameter_types
        elif not isinstance(parameter_types, ParameterType):
            raise TypeError(parameter_types, "must be of type ParameterType")
        count = None
        index = None
        if tracker is None:
            start_index = 0
        else:
            start_index = tracker.start_index
        for property_name, prop in self.__iter_parameter_group_properties():
            group_is_linked = prop.propagate
            if linked is not None:
                if group_is_linked is not linked:
                    continue

            if not (parameter_types & prop.TYPE):
                continue

            count = prop.fcount(self)
            if not count:
                continue

            stop_index = start_index + count
            if count > 1:
                index = slice(start_index, stop_index)
            elif count == 1:
                index = start_index
            else:
                index = None

            instance_key = self._linked_instance_to_key
            if group_is_linked or instance_key is None:
                name = property_name
            else:
                name = f"{instance_key}:{property_name}"

            group = ParameterGroupId(
                name=name,
                type=prop.TYPE,
                linked=group_is_linked,
                property_name=property_name,
                instance_key=instance_key,
                count=count,
                start_index=start_index,
                stop_index=stop_index,
                index=index,
                get_constraints=self.__constraints_getter(prop),
            )
            if tracker is None:
                yield group
                start_index += count
            elif tracker.is_new_group(group):
                yield group
                start_index = tracker.start_index

    def __constraints_getter(self, prop):
        def get_constraints():
            return prop.fconstraints(self)

        return get_constraints

    @linked_property
    def parameter_types(self):
        return self.__parameter_types

    @parameter_types.setter
    def parameter_types(self, value):
        if not isinstance(value, ParameterType):
            raise TypeError(value, "must be None or ParameterType")
        self.__parameter_types = value

    def _get_noncached_property_value(self, group):
        return getattr(self, group.property_name)

    def _set_noncached_property_value(self, group, value):
        setattr(self, group.property_name, value)


class _IterGroupTracker:
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


class ParameterModelManager(ParameterModelBase, LinkedModelManager):
    """Model that manages linked models that implement fit parameters."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._enable_property_link("parameter_types")
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
    def parameter_types(self):
        return self._get_linked_property_value("parameter_types")

    @parameter_types.setter
    def parameter_types(self, value):
        self._set_linked_property_value("parameter_types", value)

    def _instance_cached_property_ids(self, **paramtype):
        """
        :yields ParameterGroupId:
        """
        # Shared parameters
        tracker = _IterGroupTracker()
        for model in self.models:
            yield from model._instance_cached_property_ids(
                linked=True, tracker=tracker, **paramtype
            )
        # Non-shared parameters
        for model in self.models:
            yield from model._instance_cached_property_ids(
                linked=False, tracker=tracker, **paramtype
            )

    def _get_noncached_property_value(self, group):
        instance = self._linked_key_to_instance(group.instance_key)
        return getattr(instance, group.property_name)

    def _set_noncached_property_value(self, group, value):
        instance = self._linked_key_to_instance(group.instance_key)
        setattr(instance, group.property_name, value)

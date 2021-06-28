from contextlib import contextmanager
import numpy
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModel
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModelContainer
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

    def _property_cache_key(self, linear=None):
        if linear is None:
            linear = self.linear
        return linear

    def _create_empty_cache(self, key, **paramtype):
        return numpy.zeros(self.get_n_parameters(**paramtype))

    def _property_cache_index(self, group_name, **paramtype):
        return self._parameter_group_index(group_name, **paramtype)

    def _parameter_group_index(self, group_name, **paramtype):
        """Parameter group index in the parameter sequence

        :returns int or slice or None:
        """
        for name, idx in self._parameter_group_indices(**paramtype):
            if group_name == name:
                return idx
        return None

    def _parameter_group_indices(self, **paramtype):
        """Index of each parameter group in the parameter sequence

        :yields int or slice: index of parameter group in all parameters
        """
        i = 0
        for group_name, n in self._iter_parameter_group_count(**paramtype):
            if n == 1:
                yield group_name, i
            else:
                yield group_name, slice(i, i + n)
            i += n

    def get_parameter_group_names(self, **paramtype):
        return tuple(self._iter_parameter_group_names(**paramtype))

    def get_parameter_names(self, **paramtype):
        return tuple(self._iter_parameter_names(**paramtype))

    def get_n_parameters(self, **paramtype):
        return sum(n for _, n in self._iter_parameter_group_count(**paramtype))

    def _iter_parameter_names(self, **paramtype):
        for group_name, n in self._iter_parameter_group_count(**paramtype):
            if n > 1:
                for i in range(n):
                    yield group_name + str(i)
            else:
                yield group_name

    def _iter_parameter_group_count(self, **paramtype):
        """Yield name and count of enabled parameter groups

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        group_names = self._iter_parameter_group_names(**paramtype)
        for group_name in group_names:
            n = self._get_parameter_group_count(group_name)
            if n:
                yield group_name, n

    def _get_parameter_group_count(self, group_name):
        """
        :yield parameter_group:
        """
        raise NotImplementedError

    def _iter_parameter_group_names(self, **paramtype):
        """
        :yield str:
        """
        raise NotImplementedError


class ParameterModel(ParameterModelBase, LinkedModel):
    _PARAMETER_GROUP_NAMES = tuple()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        allp = list()
        for group_name in sorted(dir(cls)):  # TODO: how to keep order of declaration?
            attr = getattr(cls, group_name)
            if isinstance(attr, parameter_group):
                allp.append(group_name)
        cls._PARAMETER_GROUP_NAMES = tuple(allp)

    def _iter_parameter_group_names(self, linear=None, linked=None):
        for group_name in self._PARAMETER_GROUP_NAMES:
            if linked is not None:
                if self._property_is_linked(group_name) is not linked:
                    continue
            if linear is None:
                linear = self.linear
            if linear:
                if self._parameter_group_is_linear(group_name):
                    yield group_name
            else:
                yield group_name

    def _get_parameter_group_count(self, group_name):
        """
        :returns int:
        """
        return getattr(type(self), group_name).fcount(self)

    @classmethod
    def _parameter_group_is_linear(cls, group_name):
        return isinstance(getattr(cls, group_name, None), linear_parameter_group)

    @linked_property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, value):
        self._linear = value


class ParameterModelContainer(ParameterModelBase, LinkedModelContainer):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._enable_property_link("linear")
        for model in self.models:
            model._cache_manager = self

    @property
    def models(self):
        return self._linked_instances

    @property
    def nmodels(self):
        return len(self._linked_instances)

    @property
    def linear(self):
        return self._get_linked_property_value("linear")

    @linear.setter
    def linear(self, value):
        self._set_linked_property_value("linear", value)

    def _iter_parameter_group_names(self, **paramtype):
        """
        :yield str:
        """
        # Shared parameters
        encountered = set()
        for i, model in enumerate(self.models):
            for group_name in model._iter_parameter_group_names(
                linked=True, **paramtype
            ):
                if group_name in encountered:
                    continue
                encountered.add(group_name)
                yield f"model{i}:{group_name}"
        # Non-shared parameters
        for i, model in enumerate(self.models):
            for group_name in model._iter_parameter_group_names(
                linked=False, **paramtype
            ):
                yield f"model{i}:{group_name}"

    def _get_parameter_group_count(self, group_name):
        """
        :returns int:
        """
        model_name, group_name = group_name.split(":")
        i = int(model_name.replace("model", ""))
        model = self.models[i]
        return model._get_parameter_group_count(group_name)

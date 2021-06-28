from contextlib import contextmanager
import numpy
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModel
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModelContainer
from PyMca5.PyMcaMath.fitting.LinkedModel import linked_property
from PyMca5.PyMcaMath.fitting.LinkedModel import linked_contextmanager


class parameter_group(linked_property):
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


class ModelParameterInterfaceBase:
    def __init__(self):
        self.__cache = dict()
        self.linear = False
        super().__init__()

    @linked_contextmanager
    def _cachingContext(self, cachename):
        reset = not self._cachingEnabled(cachename)
        if reset:
            self.__cache[cachename] = dict()
        try:
            yield
        finally:
            if reset:
                del self.__cache[cachename]

    def _cachingEnabled(self, cachename):
        return cachename in self.__cache

    def _getCache(self, cachename, *subnames):
        if cachename in self.__cache:
            ret = self.__cache[cachename]
            for cachename in subnames:
                try:
                    ret = ret[cachename]
                except KeyError:
                    ret[cachename] = dict()
                    ret = ret[cachename]
            return ret
        else:
            return None

    @contextmanager
    def linear_context(self, linear):
        keep = self.linear
        self.linear = linear
        try:
            yield
        finally:
            self.linear = keep

    @classmethod
    def parameter_group_is_linear(cls, name):
        return isinstance(getattr(cls, name, None), linear_parameter_group)

    def get_parameter_group_names(self, **paramtype):
        return tuple(self.iter_parameter_group_names(**paramtype))

    def get_parameter_names(self, **paramtype):
        return tuple(self.iter_parameter_names(**paramtype))

    def get_n_parameters(self, **paramtype):
        return sum(n for _, n in self.iter_parameter_groups(**paramtype))

    def iter_parameter_names(self, **paramtype):
        for group_name, n in self.iter_parameter_groups(**paramtype):
            if n > 1:
                for i in range(n):
                    yield group_name + str(i)
            else:
                yield group_name

    def iter_parameter_groups(self, **paramtype):
        """Yield name and count of enabled parameter groups

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        cache = self._getCache("iter_parameter_groups")
        if cache is None:
            yield from self._parameter_groups_notcached(**paramtype)
            return

        key = self._parameters_cache_key(**paramtype)
        it = cache.get(key)
        if it is None:
            it = cache[key] = list(self._parameter_groups_notcached(**paramtype))
        yield from it

    def _parameter_groups_notcached(self, **paramtype):
        """Helper for `iter_parameter_groups`.

        :yields str, int: group name, nb. parameters in the group
        """
        names = self.iter_parameter_group_names(**paramtype)
        for name in names:
            paramprop = getattr(self.__class__, name)
            n = paramprop.fcount(self)
            if n:
                yield name, n

    def _parameters_cache_key(self, linear=None, linked=None):
        if linear is None:
            linear = self.linear
        return linear, linked

    def get_parameter_values(self, **paramtype):
        """All parameters values in one numpy array

        :returns array:
        """
        raise NotImplementedError

    def set_parameter_values(self, values, **paramtype):
        """
        :returns array:
        """
        raise NotImplementedError

    def iter_parameter_group_names(self, **paramtype):
        """
        :yield str:
        """
        raise NotImplementedError


class ModelParameterInterface(LinkedModel, ModelParameterInterfaceBase):
    _PARAMETERS = tuple()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        allp = list()
        for name in sorted(dir(cls)):  # TODO: keep order if declaration?
            attr = getattr(cls, name)
            if isinstance(attr, parameter_group):
                allp.append(name)
        cls._PARAMETERS = tuple(allp)

    @linked_property
    def linear(self):
        return self.__linear

    @linear.setter
    def linear(self, value):
        self.__linear = value

    def get_parameter_values(self, **paramtype):
        """All parameters values in one numpy array

        :returns array:
        """
        cache = self._getCache("_parameters")
        if cache is None:
            return self._get_parameter_values_notcached(**paramtype)

        key = self._parameters_cache_key()
        parameters = cache.get(key, None)
        if parameters is None:
            parameters = cache[key] = self._get_parameter_values_notcached(**paramtype)
        return parameters

    def set_parameter_values(self, values, **paramtype):
        """
        :returns array:
        """
        cache = self._getCache("_parameters")
        if cache is None:
            self._set_parameter_values_notcached(values, **paramtype)
        else:
            key = self._parameters_cache_key(**paramtype)
            cache[key] = values

    def _get_parameter_values_notcached(self, **paramtype):
        """Merge all parameters values in one numpy array

        :returns array:
        """
        nvalues = self._n_parameters(**paramtype)
        values = numpy.zeros(nvalues)
        for group_name, idx in self._parameter_group_indices(**paramtype):
            values[idx] = getattr(self, group_name)
        return values

    def _set_parameter_values_notcached(self, values, **paramtype):
        for group_name, idx in self._parameter_group_indices(**paramtype):
            setattr(self, group_name, values[idx])

    def _get_parameter(self, fget):
        parameters = self._getCache("_parameters")
        if parameters is None:
            return fget(self)

        key = self._parameters_cache_key()
        parameters = parameters.get(key, None)
        if parameters is None:
            return fget(self)

        idx = self._parameter_group_index(fget.__name__)
        if idx is None:
            return fget(self)
        return parameters[idx]

    def _set_parameter(self, fset, value):
        parameters = self._getCache("_parameters")
        if parameters is None:
            return fset(self, value)

        key = self._parameters_cache_key()
        parameters = parameters.get(key, None)
        if parameters is None:
            return fset(self, value)

        idx = self._parameter_group_index(fset.__name__)
        if idx is None:
            return fset(self, value)
        parameters[idx] = value

    def _parameter_group_index(self, name, **paramtype):
        """Parameter group index in the parameter sequence

        :returns int or slice or None:
        """
        for group_name, idx in self._parameter_group_indices(**paramtype):
            if name == group_name:
                return idx
        return None

    def _parameter_group_indices(self, **paramtype):
        """Index of each parameter group in the parameter sequence

        :yields int or slice: index of parameter group in all parameters
        """
        i = 0
        for group_name, n in self.iter_parameter_groups(**paramtype):
            if n == 1:
                yield group_name, i
            else:
                yield group_name, slice(i, i + n)
            i += n

    def iter_parameter_group_names(self, linear=None):
        for name in self._PARAMETERS:
            if linear is not None:
                is_linear = self.parameter_group_is_linear(name)
                if linear == is_linear:
                    yield name


class ConcatModelParameterInterface(LinkedModelContainer, ModelParameterInterfaceBase):
    @property
    def models(self):
        return self._linked_instances

    @property
    def nmodels(self):
        return len(self._linked_instances)

    def iter_parameter_group_names(self, **paramtype):
        """
        :yield str:
        """
        encountered = set()
        for i, instance in enumerate(self._linked_instances):
            for name in instance.iter_parameter_group_names(**paramtype):
                if instance._property_is_linked(name):
                    if name not in encountered:
                        encountered.add(name)
                        yield name
                else:
                    yield f"model{i}_{name}"

    def get_parameter_values(self, **paramtype):
        """All parameters values in one numpy array

        :returns array:
        """
        values = list()
        for instance in self._linked_instances:
            ivalues = instance.get_parameter_values(**paramtype)
            values.append(ivalues)
        return numpy.concatenate(values)

    def set_parameter_values(self, values, **paramtype):
        """
        :returns array:
        """
        i = 0
        for instance in self._linked_instances:
            n = instance.get_n_parameters(**paramtype)
            instance.set_parameter_values(values[i : i + n], **paramtype)
            i += n

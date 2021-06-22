import functools
from contextlib import ExitStack, contextmanager
from PyMca5.PyMcaMath.fitting.PropertyUtils import wrapped_property


class linked_property(wrapped_property):
    """Setting a linked property of one object
    will set that property for all linked objects
    """
    def _wrap_setter(self, fset):
        propname = fset.__name__
        fset = super()._wrap_setter(fset)

        @functools.wraps(fset)
        def wrapper(oself, value):
            ret = fset(oself, value)
            if oself.propagation_is_enabled(propname):
                for instance in oself._filter_class_has_linked_property(
                    oself._non_propagating_instances, propname
                ):
                    setattr(instance, propname, value)
            return ret

        return wrapper


def linked_contextmanager(method):
    """Entering the context manager of one object
    will enter the context manager of linked objects
    """
    context_name = method.__name__
    ctxmethod = contextmanager(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kw):
        with ExitStack() as stack:
            ctx = ctxmethod(self, *args, **kw)
            stack.enter_context(ctx)
            for instance in self._non_propagating_instances:
                ctxmgr = getattr(instance, context_name)
                if ctxmgr is not None:
                    ctx = ctxmgr(*args, **kw)
                    stack.enter_context(ctx)
            yield

    return contextmanager(wrapper)


class LinkedInterface:
    """Every class that uses the link decorators needs
    to derived from this class.
    """
    def __init__(self):
        self.__enabled_linked_properties = dict()
        self.__linked_instances = list()
        self.__propagate = True
        super().__init__()

    def propagation_is_enabled(self, name):
        if not self.__linked_instances:
            return False
        return self.property_is_linked(name)
    
    def property_is_linked(self, name):
        return self.__enabled_linked_properties.get(name, False)

    def disable_property_link(self, *names):
        for name in names:
            if self.has_linked_property(name):
                self.__enabled_linked_properties[name] = False

    def enable_property_link(self, *names):
        for name in names:
            if self.has_linked_property(name):
                self.__enabled_linked_properties[name] = True

    @property
    def linked_instances(self):
        return self.__linked_instances

    @linked_instances.setter
    def linked_instances(self, instances):
        self._propagated_linked_instances_setter(instances)

    def _propagated_linked_instances_setter(self, instances):
        others = list()
        for instance in instances:
            if instance is self:
                continue
            if not isinstance(instance, LinkedInterface):
                raise TypeError(type(instance), "can only link objects of the 'LinkedInterface' type")
            others.append(instance)
        self.__linked_instances = others
        for instance in others:
            instance._unpropagated_linked_instances_setter(instances)

    def _unpropagated_linked_instances_setter(self, instances):
        self.__linked_instances = [i for i in instances if i is not self]

    @property
    def _non_propagating_instances(self):
        if not self.__propagate:
            return
        for instance in self.linked_instances:
            with instance._disable_propagation():
                yield instance

    @contextmanager
    def _disable_propagation(self):
        keep = self.__propagate
        self.__propagate = False
        try:
            yield
        finally:
            self.__propagate = keep

    @classmethod
    def has_linked_property(cls, prop_name):
        prop = getattr(cls, prop_name, None)
        return isinstance(prop, linked_property)

    @staticmethod
    def _filter_class_has_linked_property(instances, prop_name):
        for instance in instances:
            if instance.has_linked_property(prop_name):
                yield instance


class LinkedContainerInterface:
    """Classes that manage LinkedInterface objects should
    derive from this class.
    """
    def __init__(self, linked_instances):
        self.linked_instances = linked_instances
        super().__init__()

    @property
    def linked_instances(self):
        return self.__linked_instances

    @linked_instances.setter
    def linked_instances(self, linked_instances):
        linked_instances[0].linked_instances = linked_instances
        self.__linked_instances = linked_instances

    def instances_with_linked_property(self, prop_name):
        yield from LinkedInterface._filter_class_has_linked_property(self.linked_instances, prop_name)

    def instance_with_linked_property(self, prop_name):
        for instance in self.instances_with_linked_property(prop_name):
            return instance
        return None

    def get_linked_property(self, prop_name):
        instance = self.instance_with_linked_property(prop_name)
        if instance is None:
            raise ValueError(f"No instance has linked property {repr(prop_name)}")
        return getattr(instance, prop_name)

    def set_linked_property(self, prop_name, value):
        instance = self.instance_with_linked_property(prop_name)
        if instance is None:
            raise ValueError(f"No instance has linked property {repr(prop_name)}")
        setattr(instance, prop_name, value)

    def disable_property_link(self, *names):
        for name in names:
            for instance in self.instances_with_linked_property(name):
                instance.disable_property_link(name)

    def enable_property_link(self, *names):
        for name in names:
            value = self.get_linked_property(name)
            for i, instance in enumerate(self.instances_with_linked_property(name)):
                instance.enable_property_link(name)
            self.set_linked_property(name, value)

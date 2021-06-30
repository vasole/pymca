import functools
from contextlib import ExitStack, contextmanager
from collections.abc import Mapping
from PyMca5.PyMcaMath.fitting.model.PropertyUtils import wrapped_property


class linked_property(wrapped_property):
    """Setting a linked property of one object
    will set that property for all linked objects
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.propagate = False

    def _wrap_setter(self, fset):
        propname = fset.__name__
        fset = super()._wrap_setter(fset)

        @functools.wraps(fset)
        def wrapper(oself, value):
            fset(oself, value)
            if not self.propagate:
                return
            for instance in oself._filter_class_has_linked_property(
                oself._non_propagating_instances, propname
            ):
                setattr(instance, propname, value)

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


class LinkedModel:
    """Model with properties and context's that are linked to other
    LinkedModel instances.
    """

    def __init__(self, *args, **kw):
        self._link_manager = None
        self.__propagate = True
        super().__init__(*args, **kw)

    @classmethod
    def _get_linked_property(cls, prop_name):
        prop = getattr(cls, prop_name, None)
        if isinstance(prop, linked_property):
            return prop
        return None

    @classmethod
    def _has_linked_property(cls, prop_name):
        return cls._get_linked_property(prop_name) is not None

    @classmethod
    def _property_is_linked(cls, prop_name):
        prop = cls._get_linked_property(prop_name)
        if prop is None:
            return None
        return prop.propagate

    @classmethod
    def _disable_property_link(cls, *prop_names):
        for prop_name in prop_names:
            prop = cls._get_linked_property(prop_name)
            if prop is not None:
                prop.propagate = False

    @classmethod
    def _enable_property_link(cls, *prop_names):
        for prop_name in prop_names:
            prop = cls._get_linked_property(prop_name)
            if prop is not None:
                prop.propagate = True

    @property
    def _link_manager(self):
        return self.__link_manager

    @_link_manager.setter
    def _link_manager(self, obj):
        if obj is not None and not isinstance(obj, LinkedModelManager):
            raise TypeError(obj, type(obj))
        self.__link_manager = obj

    @property
    def _linked_instances(self):
        if self._link_manager is None:
            return
        for instance in self._link_manager._linked_instances:
            if instance is not self:
                yield instance

    @property
    def _linked_instance_to_key(self):
        if self._link_manager is None:
            return None
        return self._link_manager._linked_instance_to_key(self)

    @property
    def _non_propagating_instances(self):
        if not self.__propagate:
            return
        for instance in self._linked_instances:
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

    @staticmethod
    def _filter_class_has_linked_property(instances, prop_name):
        for instance in instances:
            if instance._has_linked_property(prop_name):
                yield instance


class LinkedModelManager:
    """Model that manages linked LinkedModel objects"""

    def __init__(self, linked_instances=None, *args, **kw):
        super().__init__(*args, **kw)
        self._linked_instance_mapping = linked_instances

    @property
    def _linked_instances(self):
        return self.__linked_instance_mapping.values()

    @property
    def _linked_instance_mapping(self):
        return self.__linked_instance_mapping

    @_linked_instance_mapping.setter
    def _linked_instance_mapping(self, instances):
        if instances is None:
            instances = dict()
        elif not isinstance(instances, Mapping):
            raise TypeError(instances, "Linked instance must be a 'Mapping'")
        for instance in instances.values():
            if not isinstance(instance, LinkedModel):
                raise TypeError(
                    instance, "Linked instance must be of type 'LinkedModel'"
                )
        self.__linked_instance_mapping = instances
        for instance in instances.values():
            instance._link_manager = self

    def _linked_instance_to_key(self, instance):
        for name, _instance in self._linked_instance_mapping.items():
            if _instance is instance:
                return name
        return None

    def _linked_key_to_instance(self, name):
        return self._linked_instance_mapping[name]

    def _instances_with_linked_property(self, prop_name):
        yield from LinkedModel._filter_class_has_linked_property(
            self._linked_instances, prop_name
        )

    def _instance_with_linked_property(self, prop_name):
        for instance in self._instances_with_linked_property(prop_name):
            return instance
        return None

    def _get_linked_property(self, prop_name):
        instance = self._instance_with_linked_property(prop_name)
        if instance is None:
            raise ValueError(f"No instance has linked property {repr(prop_name)}")
        return getattr(type(instance), prop_name)

    def _get_linked_property_value(self, prop_name):
        instance = self._instance_with_linked_property(prop_name)
        if instance is None:
            raise ValueError(f"No instance has linked property {repr(prop_name)}")
        return getattr(instance, prop_name)

    def _set_linked_property_value(self, prop_name, value):
        instance = self._instance_with_linked_property(prop_name)
        if instance is None:
            raise ValueError(f"No instance has linked property {repr(prop_name)}")
        setattr(instance, prop_name, value)

    def _disable_property_link(self, *names):
        for name in names:
            for instance in self._instances_with_linked_property(name):
                instance._disable_property_link(name)

    def _enable_property_link(self, *names):
        for name in names:
            value = self._get_linked_property_value(name)
            for instance in self._instances_with_linked_property(name):
                instance._enable_property_link(name)
            self._set_linked_property_value(name, value)

class _wrapped_property(property):
    """Property that prepares fget, fset and fdel wrappers
    for derived property classes.
    """

    def __init__(
        self,
        fget=None,
        fset=None,
        fdel=None,
        doc=None,
    ) -> None:
        if fget is not None:
            fget = self._wrap_getter(fget)
        if fset is not None:
            fset = self._wrap_setter(fset)
        if fdel is not None:
            fget = self._wrap_deleter(fdel)
        super().__init__(
            fget=fget,
            fset=fset,
            fdel=fdel,
            doc=doc,
        )

    def getter(self, fget):
        """Decorator to change fget after property instantiation"""
        if fget is not None:
            fget = self._wrap_getter(fget)
        return super().getter(fget)

    def setter(self, fset):
        """Decorator to change fset after property instantiation"""
        if fset is not None:
            fset = self._wrap_setter(fset)
        return super().setter(fset)

    def deleter(self, fdel):
        """Decorator to change fdel after property instantiation"""
        if fdel is not None:
            fdel = self._wrap_deleter(fdel)
        return super().deleter(fdel)

    def _wrap_getter(self, fget):
        """Intended for derived property classes"""
        return fget

    def _wrap_setter(self, fset):
        """Intended for derived property classes"""
        return fset

    def _wrap_deleter(self, fdel):
        """Intended for derived property classes"""
        return fdel


class wrapped_property:
    """Can be used like python's builtin property but with
    getter and setter hooks for derived classes.
    """

    def __init__(self, fget):
        self.fget = self._wrap_getter(fget)
        self.fset = None
        self.attrname = None

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        else:
            raise TypeError(f"Cannot use the same {type(self).__name__} twice")

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError(
                f"{type(instance).__name__}.{self.attrname} has no setter"
            )
        if instance is None:
            return self
        return self.fset(instance, value)

    def setter(self, fset):
        self.fset = self._wrap_setter(fset)
        return self

    def _wrap_getter(self, fget):
        """Intended for derived property classes"""
        return fget

    def _wrap_setter(self, fset):
        """Intended for derived property classes"""
        return fset

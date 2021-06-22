import functools
from contextlib import contextmanager
from PyMca5.PyMcaMath.fitting.PropertyUtils import wrapped_property


class cached_property(wrapped_property):
    def _wrap_getter(self, fget):
        fget = super()._wrap_getter(fget)

        @functools.wraps(fget)
        def wrapper(oself):
            return oself._cached_property_fget(fget)

        return wrapper

    def _wrap_setter(self, fset):
        fset = super()._wrap_setter(fset)

        @functools.wraps(fset)
        def wrapper(oself, value):
            return oself._cached_property_fset(fset, value)

        return wrapper


class CachedInterface:
    _CACHED_PROPERTIES = tuple()

    def __init_subclass__(subcls, **kwargs):
        super().__init_subclass__(**kwargs)
        allp = list()
        for name, attr in vars(subcls).items():
            if isinstance(attr, cached_property):
                allp.append(name)
        subcls._CACHED_PROPERTIES = subcls._CACHED_PROPERTIES + tuple(allp)

    @classmethod
    def _cached_properties(self):
        return self._CACHED_PROPERTIES

    def __init__(self):
        self._cache_object = None
        self._cache_object._cache_root = dict()
        super().__init__()

    @property
    def _cache_object(self):
        if self.__external_cache_object is None:
            return self
        else:
            return self.__external_cache_object

    @_cache_object.setter
    def _cache_object(self, obj):
        if obj is not None and not isinstance(obj, CachedInterface):
            raise TypeError(obj, type(obj))
        self.__external_cache_object = obj

    @contextmanager
    def cachingContext(self, cachename):
        cache_root = self._cache_object._cache_root
        new_context_entry = cachename not in cache_root
        if new_context_entry:
            cache_root[cachename] = dict()
        try:
            yield cache_root[cachename]
        finally:
            if new_context_entry:
                del cache_root[cachename]

    def cachingEnabled(self, cachename):
        return cachename in self._cache_object._cache_root

    def getCache(self, cachename, *subnames):
        cache_root = self._cache_object._cache_root
        if cachename not in cache_root:
            return None
        ret = cache_root[cachename]
        for cachename in subnames:
            try:
                ret = ret[cachename]
            except KeyError:
                ret[cachename] = dict()
                ret = ret[cachename]
        return ret

    @contextmanager
    def propertyCachingContext(self, persist=False, start_cache=None, **cacheoptions):
        values_cache = self._get_property_values_cache(**cacheoptions)
        if values_cache is not None:
            # Re-entering this context should not affect anything
            yield values_cache
            return

        if start_cache is None:
            # Fill and empty cache with property values
            cache_object = self._cache_object
            key = cache_object._property_cache_key(**cacheoptions)
            values_cache = cache_object._create_empty_cache(key, **cacheoptions)
            nameindexmap = dict()
            for name in self._cached_properties():
                index = cache_object._property_cache_index(name)
                nameindexmap[name] = index
                values_cache[index] = getattr(self, name)
        else:
            values_cache = start_cache
            nameindexmap = None

        with self.cachingContext("_cached_properties"):
            # Initialize the property values cache
            self._set_property_values_cache(values_cache, **cacheoptions)
            yield values_cache

        if persist:
            # Set property values to the cached values
            if nameindexmap is None:
                cache_object = self._cache_object
                for name in self._cached_properties():
                    index = cache_object._property_cache_index(name)
                    setattr(self, name, values_cache[index])
            else:
                for name, index in nameindexmap.items():
                    setattr(self, name, values_cache[index])

    def _get_property_values_cache(self, **cacheoptions):
        caches = self.getCache("_cached_properties")
        if caches is None:
            return None
        key = self._cache_object._property_cache_key(**cacheoptions)
        return caches.get(key, None)

    def _set_property_values_cache(self, values_cache, **cacheoptions):
        caches = self.getCache("_cached_properties")
        if caches is None:
            return
        cache_object = self._cache_object
        key = cache_object._property_cache_key(**cacheoptions)
        caches[key] = values_cache

    def _cached_property_fget(self, fget):
        values_cache = self._get_property_values_cache()
        if values_cache is None:
            return fget(self)
        index = self._cache_object._property_cache_index(fget.__name__)
        return values_cache[index]

    def _cached_property_fset(self, fset, value):
        values_cache = self._get_property_values_cache()
        if values_cache is None:
            return fset(self, value)
        index = self._cache_object._property_cache_index(fset.__name__)
        values_cache[index] = value

    def _create_empty_cache(self, key, **cacheoptions):
        # By default the property cache is a dictionary
        return dict()

    def _property_cache_index(self, name):
        # By default the property cache index is its name
        return name

    def _property_cache_key(self, **cacheoptions):
        # By default we only manage 1 cache (None)
        return None

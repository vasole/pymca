import functools
from contextlib import contextmanager
from PyMca5.PyMcaMath.fitting.PropertyUtils import wrapped_property


class CacheManager:
    """Object that manages a cache"""

    def __init__(self):
        self._cache_root = dict()
        super().__init__()

    def _create_empty_cache(self, key, **cacheoptions):
        # By default the property cache is a dictionary
        return dict()

    def _property_cache_index(self, name):
        # By default the property cache index is its name
        return name

    def _property_cache_key(self, **cacheoptions):
        # By default we only manage 1 cache (None)
        return None


class CachingModel(CacheManager):
    """Object that manages and uses an internal cache (default) or
    uses an external cache.
    """

    def __init__(self):
        self._cache_manager = None
        super().__init__()

    @property
    def _cache_manager(self):
        if self.__external_cache_manager is None:
            return self
        else:
            return self.__external_cache_manager

    @_cache_manager.setter
    def _cache_manager(self, obj):
        if obj is not None and not isinstance(obj, CacheManager):
            raise TypeError(obj, type(obj))
        self.__external_cache_manager = obj

    @contextmanager
    def _cachingContext(self, cachename):
        cache_root = self._cache_manager._cache_root
        new_context_entry = cachename not in cache_root
        if new_context_entry:
            cache_root[cachename] = dict()
        try:
            yield cache_root[cachename]
        finally:
            if new_context_entry:
                del cache_root[cachename]

    def _cachingEnabled(self, cachename):
        return cachename in self._cache_manager._cache_root

    def _getCache(self, cachename, *subnames):
        cache_root = self._cache_manager._cache_root
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


class CachedPropertiesModel(CachingModel):
    """Object with cached properties when enabled."""

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

    @contextmanager
    def _propertyCachingContext(self, persist=False, start_cache=None, **cacheoptions):
        values_cache = self._get_property_values_cache(**cacheoptions)
        if values_cache is not None:
            # Re-entering this context should not affect anything
            yield values_cache
            return

        if start_cache is None:
            # Fill and empty cache with property values
            _cache_manager = self._cache_manager
            key = _cache_manager._property_cache_key(**cacheoptions)
            values_cache = _cache_manager._create_empty_cache(key, **cacheoptions)
            nameindexmap = dict()
            for name in self._cached_properties():
                index = _cache_manager._property_cache_index(name)
                nameindexmap[name] = index
                values_cache[index] = getattr(self, name)
        else:
            values_cache = start_cache
            nameindexmap = None

        with self._cachingContext("_cached_properties"):
            # Initialize the property values cache
            self._set_property_values_cache(values_cache, **cacheoptions)
            yield values_cache

        if persist:
            # Set property values to the cached values
            if nameindexmap is None:
                _cache_manager = self._cache_manager
                for name in self._cached_properties():
                    index = _cache_manager._property_cache_index(name)
                    setattr(self, name, values_cache[index])
            else:
                for name, index in nameindexmap.items():
                    setattr(self, name, values_cache[index])

    def _get_property_values_cache(self, **cacheoptions):
        caches = self._getCache("_cached_properties")
        if caches is None:
            return None
        key = self._cache_manager._property_cache_key(**cacheoptions)
        return caches.get(key, None)

    def _set_property_values_cache(self, values_cache, **cacheoptions):
        caches = self._getCache("_cached_properties")
        if caches is None:
            return
        _cache_manager = self._cache_manager
        key = _cache_manager._property_cache_key(**cacheoptions)
        caches[key] = values_cache

    def _cached_property_fget(self, fget):
        values_cache = self._get_property_values_cache()
        if values_cache is None:
            return fget(self)
        index = self._cache_manager._property_cache_index(fget.__name__)
        return values_cache[index]

    def _cached_property_fset(self, fset, value):
        values_cache = self._get_property_values_cache()
        if values_cache is None:
            return fset(self, value)
        index = self._cache_manager._property_cache_index(fset.__name__)
        values_cache[index] = value

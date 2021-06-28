import functools
from contextlib import contextmanager
from PyMca5.PyMcaMath.fitting.PropertyUtils import wrapped_property


class CacheManager:
    """Object that manages a cache"""

    def __init__(self, *args, **kw):
        self._cache_root = dict()
        super().__init__(*args, **kw)

    def _create_empty_property_values_cache(self, key, **cacheoptions):
        # By default the property cache is a dictionary
        return dict()

    def _property_cache_index(self, name, **cacheoptions):
        # By default the property cache index is its name
        return name

    def _property_cache_key(self, **cacheoptions):
        # By default we only manage 1 cache (None)
        return None


class CachingModel(CacheManager):
    """Object that manages and uses an internal cache (default) or
    uses an external cache.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._cache_manager = None

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

        with self._cachingContext("_cached_property_indices"):
            if start_cache is None:
                values_cache = self._create_start_property_values_cache(**cacheoptions)
            else:
                values_cache = start_cache

            with self._cachingContext("_cached_properties"):
                self._set_property_values_cache(values_cache, **cacheoptions)
                yield values_cache

            if persist:
                self._persist_property_values(values_cache, **cacheoptions)

    def _get_property_values_cache(self, **cacheoptions):
        caches = self._getCache("_cached_properties")
        if caches is None:
            return None
        key = self._cache_manager._property_cache_key(**cacheoptions)
        return caches.get(key, None)

    def _set_property_values_cache(self, values_cache, **cacheoptions):
        caches = self._getCache("_cached_properties")
        if caches is None:
            return False
        _cache_manager = self._cache_manager
        key = _cache_manager._property_cache_key(**cacheoptions)
        caches[key] = values_cache
        return True

    def _create_start_property_values_cache(self, **cacheoptions):
        # Fill and empty cache with property values
        _cache_manager = self._cache_manager
        key = _cache_manager._property_cache_key(**cacheoptions)
        values_cache = _cache_manager._create_empty_property_values_cache(
            key, **cacheoptions
        )
        for name in self._cached_properties():
            index = self._get_property_cache_index(name, **cacheoptions)
            values_cache[index] = getattr(self, name)
        return values_cache

    def _get_property_values(self, **cacheoptions):
        values = self._get_property_values_cache(**cacheoptions)
        if values is None:
            return self._create_start_property_values_cache(**cacheoptions)
        return values

    def _set_property_values(self, values, **cacheoptions):
        success = self._set_property_values_cache(values, **cacheoptions)
        if not success:
            self._persist_property_values(values)

    def _persist_property_values(self, values, **cacheoptions):
        for name in self._cached_properties():
            index = self._get_property_cache_index(name, **cacheoptions)
            setattr(self, name, values[index])

    def _get_property_indices_cache(self, **cacheoptions):
        caches = self._getCache("_cached_property_indices")
        if caches is None:
            return None
        key = self._cache_manager._property_cache_key(**cacheoptions)
        return caches.get(key, None)

    def _cached_property_fget(self, fget):
        values_cache = self._get_property_values_cache()
        if values_cache is None:
            return fget(self)
        index = self._get_property_cache_index(fget.__name__)
        return values_cache[index]

    def _cached_property_fset(self, fset, value):
        values_cache = self._get_property_values_cache()
        if values_cache is None:
            return fset(self, value)
        index = self._get_property_cache_index(fset.__name__)
        values_cache[index] = value

    def _get_property_cache_index(self, name, **cacheoptions):
        name_to_index = self._get_property_indices_cache(**cacheoptions)
        if name_to_index is None:
            return self._cache_manager._property_cache_index(name, **cacheoptions)
        if name in name_to_index:
            return self.name_to_index[name]
        index = self._cache_manager._property_cache_index(name, **cacheoptions)
        self.name_to_index[name] = index
        return index

import functools
from contextlib import contextmanager
from PyMca5.PyMcaMath.fitting.model.PropertyUtils import wrapped_property


class CacheManager:
    """Object that manages a cache"""

    def __init__(self, *args, **kw):
        self._cache_root = dict()
        super().__init__(*args, **kw)

    def _create_empty_property_values_cache(self, key, **cacheoptions):
        # By default the property cache is a dictionary
        return dict()

    def _property_index_from_id(self, propid, **cacheoptions):
        # By default the property cache index is its propid
        return propid

    def _property_cache_key(self, **cacheoptions):
        # By default we only manage 1 cache (None)
        return None


class CachingModel(CacheManager):
    """Model that manages and uses an internal cache (default)
    or uses an external cache.
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
    """Property getter/setter may get/set from
    a cache when enabled.
    """

    def _wrap_getter(self, fget):
        @functools.wraps(fget)
        def wrapper(oself, nocache=False):
            if nocache:
                return fget(oself)
            else:
                return oself._cached_property_fget(fget)

        return super()._wrap_getter(wrapper)

    def _wrap_setter(self, fset):
        @functools.wraps(fset)
        def wrapper(oself, value, nocache=False):
            if nocache:
                return fset(oself, value)
            else:
                return oself._cached_property_fset(fset, value)

        return super()._wrap_setter(wrapper)


class CachedPropertiesModel(CachingModel):
    """Model that implements cached properties"""

    _CACHED_PROPERTIES = tuple()

    def __init_subclass__(subcls, **kwargs):
        super().__init_subclass__(**kwargs)
        allp = list(subcls._CACHED_PROPERTIES)
        for name, attr in vars(subcls).items():
            if isinstance(attr, cached_property) and name not in allp:
                allp.append(name)
        subcls._CACHED_PROPERTIES = tuple(sorted(allp))

    @classmethod
    def _cached_property_names(cls):
        """All property names for this class"""
        return cls._CACHED_PROPERTIES

    def _instance_cached_property_ids(self, **cacheoptions):
        """All property id's for this instance and the provided cache options"""
        return self._cached_property_names()

    @contextmanager
    def _propertyCachingContext(self, persist=False, start_cache=None, **cacheoptions):
        values_cache = self._get_property_values_cache(**cacheoptions)
        if values_cache is not None:
            # Re-entering this context should not affect anything
            yield values_cache
            return

        with self._cachingContext("_propid"):
            if start_cache is None:
                values_cache = self._create_start_property_values_cache(**cacheoptions)
            else:
                values_cache = start_cache

            with self._cachingContext("_property_values"):
                self._set_property_values_cache(values_cache, **cacheoptions)
                yield values_cache

            if persist:
                self._persist_property_values(values_cache, **cacheoptions)

    def _in_property_caching_context(self):
        return self._getCache("_property_values") is not None

    def _get_property_values_cache(self, **cacheoptions):
        caches = self._getCache("_property_values")
        if caches is None:
            return None
        key = self._cache_manager._property_cache_key(**cacheoptions)
        return caches.get(key, None)

    def _set_property_values_cache(self, values_cache, **cacheoptions):
        caches = self._getCache("_property_values")
        if caches is None:
            return False
        key = self._cache_manager._property_cache_key(**cacheoptions)
        caches[key] = values_cache
        return True

    def _create_start_property_values_cache(self, **cacheoptions):
        """Fill an empty cache with property values"""
        _cache_manager = self._cache_manager
        key = _cache_manager._property_cache_key(**cacheoptions)
        values_cache = _cache_manager._create_empty_property_values_cache(
            key, **cacheoptions
        )
        for propid in self._iter_cached_property_ids(**cacheoptions):
            index = self._propid_to_index(propid, **cacheoptions)
            values_cache[index] = self._get_noncached_property_value(propid)
        return values_cache

    def _get_property_values(self, **cacheoptions):
        """Get the cache when enabled, get instance property values when disabled"""
        values = self._get_property_values_cache(**cacheoptions)
        if values is None:
            return self._create_start_property_values_cache(**cacheoptions)
        return values

    def _set_property_values(self, values, **cacheoptions):
        """Set the cache when enabled, set instance property values when disabled"""
        success = self._set_property_values_cache(values, **cacheoptions)
        if not success:
            self._persist_property_values(values, **cacheoptions)

    def _persist_property_values(self, values, **cacheoptions):
        for propid in self._iter_cached_property_ids(**cacheoptions):
            index = self._propid_to_index(propid, **cacheoptions)
            self._set_noncached_property_value(propid, values[index])

    def _get_property_value(self, propid, **cacheoptions):
        """Get the value from the cache or from the property"""
        values_cache = self._get_property_values_cache(**cacheoptions)
        if values_cache is None:
            return self._get_noncached_property_value(propid)
        index = self._propid_to_index(propid, **cacheoptions)
        return values_cache[index]

    def _get_noncached_property_value(self, propid):
        name = self._property_name_from_id(propid)
        return getattr(self, name)

    def _set_property_value(self, propid, value, **cacheoptions):
        """Set the value in the cache or the property"""
        values_cache = self._get_property_values_cache(**cacheoptions)
        if values_cache is None:
            return self._set_noncached_property_value(propid, value)
        index = self._propid_to_index(propid, **cacheoptions)
        values_cache[index] = value

    def _set_noncached_property_value(self, propid, value):
        name = self._property_name_from_id(propid)
        setattr(self, name, value)

    def _cached_property_fget(self, fget):
        """Same as _get_property_value but we have the property object
        instead of the propid
        """
        values_cache = self._get_property_values_cache()
        propid = self._name_to_propid(fget.__name__)
        if values_cache is None or propid is None:
            return fget(self)
        index = self._propid_to_index(propid)
        return values_cache[index]

    def _cached_property_fset(self, fset, value):
        """Same as _set_property_value but we have the property object
        instead of the propid
        """
        values_cache = self._get_property_values_cache()
        propid = self._name_to_propid(fset.__name__)
        if values_cache is None or propid is None:
            return fset(self, value)
        index = self._propid_to_index(propid)
        values_cache[index] = value

    def _get_from_propid_cache(self, *subnames, dtype=dict, **cacheoptions):
        """Returns the cache when propid caching is enabled, potentially initialized
        by instantiating `dtype`. Returns `None` when propid caching is disabled.
        """
        caches = self._getCache("_propid", *subnames)
        if caches is None:
            return None
        key = self._cache_manager._property_cache_key(**cacheoptions)
        return caches.setdefault(key, dtype())

    def _iter_cached_property_ids(self, **cacheoptions):
        """To be used when iterating over all property id's
        of this instance.
        """
        propid_list = self._get_from_propid_cache(
            "_propid_list", dtype=list, **cacheoptions
        )
        if propid_list is None:
            yield from self._instance_cached_property_ids(**cacheoptions)
            return
        if not propid_list:
            propid_list.extend(self._instance_cached_property_ids(**cacheoptions))
        yield from propid_list

    def _propid_to_index(self, propid, **cacheoptions):
        propid_to_index = self._get_from_propid_cache(
            "_propid_to_index", dtype=dict, **cacheoptions
        )
        if propid_to_index is None:
            return self._cache_manager._property_index_from_id(propid, **cacheoptions)
        if propid in propid_to_index:
            return propid_to_index[propid]
        index = self._cache_manager._property_index_from_id(propid, **cacheoptions)
        propid_to_index[propid] = index
        return index

    def _name_to_propid(self, property_name, **cacheoptions):
        return self._property_id_from_name(property_name)
        # TODO: duplicate names for linked models
        name_to_propid = self._get_from_propid_cache(
            "_name_to_propid", dtype=dict, **cacheoptions
        )
        if name_to_propid is None:
            return self._property_id_from_name(property_name)
        if property_name in name_to_propid:
            return name_to_propid[property_name]
        propid = self._property_id_from_name(property_name)
        name_to_propid[property_name] = propid
        return propid

    def _property_id_from_name(self, property_name):
        return property_name

    def _property_name_from_id(self, propid):
        return propid

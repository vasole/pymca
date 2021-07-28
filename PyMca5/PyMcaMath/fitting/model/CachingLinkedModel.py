import numpy
from PyMca5.PyMcaMath.fitting.model.LinkedModel import LinkedModel
from PyMca5.PyMcaMath.fitting.model.CachingModel import CachedPropertiesModel


class CachedPropertiesLinkModel(CachedPropertiesModel, LinkedModel):
    def _iter_cached_property_ids(self, **cacheoptions):
        if self.is_linked and self._in_property_caching_context():
            # Yield only the property id's that belong to this model
            names = self._cached_property_names()
            for propid in super()._iter_cached_property_ids(**cacheoptions):
                if propid.property_name in names:
                    yield propid
        else:
            yield from super()._iter_cached_property_ids(**cacheoptions)

    def _get_property_values(self, **cacheoptions):
        values = super()._get_property_values(**cacheoptions)
        return self.__extract_values(values, **cacheoptions)

    def _set_property_values(self, values, **cacheoptions):
        values = self.__insert_values(values, **cacheoptions)
        super()._set_property_values(values, **cacheoptions)

    def __extract_values(self, values, **cacheoptions):
        if not self.is_linked:
            return values
        # Only the values that belong to this model
        data = list()
        for propid in self._iter_cached_property_ids(**cacheoptions):
            index = self._propid_to_index(propid, **cacheoptions)
            data.append(numpy.atleast_1d(values[index]).tolist())

        return numpy.concatenate(data)

    def __insert_values(self, values, **cacheoptions):
        if not self.is_linked:
            return values
        gvalues = super()._get_property_values(**cacheoptions)
        i = 0
        for propid in self._iter_cached_property_ids(**cacheoptions):
            index = self._propid_to_index(propid, **cacheoptions)
            try:
                n = len(gvalues[index])
            except TypeError:
                n = 1
            gvalues[index] = values[i : i + n]
            i += n
        return gvalues

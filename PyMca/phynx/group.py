"""
"""

from __future__ import absolute_import, with_statement

import posixpath

import h5py
import numpy as np
import re

from .base import _PhynxProperties
from .dataset import Axis, Dataset, Signal
from .exceptions import H5Error
from .registry import registry
from .utils import sync

class h5Group(h5py.Group):
    # This class is actually not needed and its methods
    # could be inside the Group class
    # It is just to make evident the changes made.

    
    #What is the use of sync?
    @sync
    def listobjects(self):
        #how to pproperly pass this list to Group?
        object_list = h5py.Group.listobjects(self)
        if self.name != "/":
            return object_list
        if (self._sorting_list is None) or\
           (len(object_list) < 2):
            return object_list

        #look for members
        sorting_key = None

        #assume all entries have the same structure
        names = object_list[0].listnames()
        for key in self._sorting_list:
            if key in names:
                sorting_key = key
                break
            
        if sorting_key is None:
            if 'name' not in self._sorting_list:
                # I could check entry attributes too.
                # ALBA was using an 'epoch' attribute at
                # at a certain point
                # Perhaps in the future ...
                return object_list
            else:
                sorting_key = 'name'

        try:
            if sorting_key != 'name':
                sorting_list = [(o[sorting_key].value, o)
                               for o in object_list]
                sorting_list.sort()
                return [x[1] for x in sorting_list]

            if sorting_key == 'name':
                sorting_list = [(self._get_number_list(o.name),o)
                               for o in object_list]
                sorting_list.sort()
                return [x[1] for x in sorting_list]
        except:
            #The only way to reach this point is to have different
            #structures among the different entries. In that case
            #defaults to the unfiltered case
            print("WARNING: Default ordering")
            return object_list

    def _get_number_list(self, txt):
        rexpr = '[/a-zA-Z:-]'
        nbs= [float(w) for w in re.split(rexpr, txt) if w not in ['',' ']]
        return nbs

    #What is the use of sync?
    @sync
    def listnames(self):
        if self.name != "/":
            return h5py.Group.listnames(self)
        object_list = self.listobjects()
        return [o.name[1:] for o in object_list]

    #What is the use of sync?
    @sync
    def iterobjects(self):
        if self.name != "/":
            return h5py.Group.listobjects(self)
        return self.listobjects()

class Group(h5Group, _PhynxProperties):

    """
    """

    @property
    def children(self):
        return self.listobjects()

    @property
    @sync
    def entry(self):
        try:
            target = self['/'.join(self.name.split('/')[:2])]
            assert isinstance(target, registry['Entry'])
            return target
        except AssertionError:
            return None

    @property
    @sync
    def measurement(self):
        try:
            return self.entry.measurement
        except AttributeError:
            return None

    @property
    def parent(self):
        return self[posixpath.split(self.name)[0]]

    @property
    @sync
    def signals(self):
        return dict(
            [(posixpath.split(s.name)[-1], s) for s in self.iterobjects()
                if isinstance(s, Signal)]
        )

    @property
    @sync
    def axes(self):
        return dict(
            [(posixpath.split(a.name)[-1], a) for a in self.iterobjects()
                if isinstance(a, Axis)]
        )

    def __init__(self, parent_object, name, create=False, **attrs):
        """
        Open an existing group or create a new one.

        attrs is a python dictionary of strings and numbers to be saved as hdf5
        attributes of the group.

        """
        if hasattr(parent_object, "_sorting_list"):
            self._sorting_list = parent_object._sorting_list
        else:
            self._sorting_list = None

        with parent_object.plock:
            h5Group.__init__(self, parent_object, name, create=create)
            if create:
                self.attrs['class'] = self.__class__.__name__
                try:
                    self.attrs['NX_class'] = self.nx_class
                except AttributeError:
                    pass

            _PhynxProperties.__init__(self, parent_object)

            if attrs:
                for attr, val in attrs.iteritems():
                    if not np.isscalar(val):
                        val = str(val)
                    self.attrs[attr] = val

    @sync
    def __repr__(self):
        try:
            return '<%s group "%s" (%d members, %d attrs)>' % (
                self.__class__.__name__,
                self.name,
                len(self),
                len(self.attrs)
            )
        except Exception:
            return "<Closed %s group>" % self.__class__.__name__

    @sync
    def __getitem__(self, name):
        # TODO: would be better to check the attribute without having to
        # create create the group twice. This might be possible with the
        # 1.8 API.
        item = super(Group, self).__getitem__(name)
        if 'class' in item.attrs:
            return registry[item.attrs['class']](self, name)
        elif 'NX_class' in item.attrs:
            return registry[item.attrs['NX_class']](self, name)
        elif isinstance(item, h5py.Dataset):
            return Dataset(self, name)
        else:
            return Group(self, name)

    @sync
    def __setitem__(self, name, value):
        super(Group, self).__setitem__(name, value)

    @sync
    def get_sorted_axes_list(self, direction=1):
        return sorted([a for a in self.axes.values() if a.axis==direction])

    @sync
    def get_sorted_signals_list(self, direction=1):
        return sorted([s for s in self.signals.values()])

    def create_dataset(self, name, *args, **kwargs):
        type = kwargs.pop('type', 'Dataset')
        return registry[type](self, name, *args, **kwargs)

    @sync
    def require_dataset(self, name, *args, **kwargs):
        type = kwargs.setdefault('type', 'Dataset')
        if not name in self:
            return self.create_dataset(name, *args, **kwargs)
        else:
            item = self[name]
            if not isinstance(item, registry[type]):
                raise NameError(
                    "Incompatible object (%s) already exists" % \
                    item.__class__.__name__
                )
            return item

    def create_group(self, name, type='Group', **data):
        return registry[type](self, name, create=True, **data)

    @sync
    def require_group(self, name, type='Group', **data):
        if not name in self:
            return self.create_group(name, type, **data)
        else:
            item = self[name]
            if not isinstance(item, registry[type]):
                raise NameError(
                    "Incompatible object (%s) already exists" % \
                    item.__class__.__name__
                )
            return item

registry.register(Group)

#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "Wout De Nolf"
__contact__ = "wout.de_nolf@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import sys
import h5py
import time
import datetime
import numpy
import os
import errno
from contextlib import contextmanager
from .. import version

if sys.version_info < (3,):
    text_dtype = h5py.special_dtype(vlen=unicode)
else:
    text_dtype = h5py.special_dtype(vlen=str)


def vlen_string(s):
    """
    Variable-length UTF-8 string (array or scalar)

    :param array(str) or str s:
    :returns numpy.ndarray:
    """
    return numpy.array(s, dtype=text_dtype)

PROGRAM_NAME = vlen_string('pymca')
PROGRAM_VERSION = vlen_string(version())
DEFAULT_PLOT_NAME = 'plotselect'


class LocalTZinfo(datetime.tzinfo):
    """
    Local timezone
    """
    _offset = datetime.timedelta(seconds=-time.timezone)
    _dst = datetime.timedelta(0)
    _name = time.tzname[time.daylight]

    def utcoffset(self, dt):
        return self.__class__._offset

    def dst(self, dt):
        return self.__class__._dst

    def tzname(self, dt):
        return self.__class__._name

localtz = LocalTZinfo()


def timestamp():
    return vlen_string(datetime.datetime.now(tz=localtz).isoformat())


def mkdir(path):
    """
    Create directory recursively when it does not exist

    :param str path:
    :raises OSError:
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise e


def split_uri(uri):
    """
    Split Uniform Resource Identifier (URI)

    :param str uri: URI
    :return tuple: filename(str), group(str)
    """
    lst = uri.split('::')
    if len(lst) == 1:
        h5groupname = ''
    else:
        h5groupname = '/'.join(lst[1:])
    return lst[0], '/' + h5groupname


def iterup(h5group, includeself=True):
    """
    Iterator which yields all parent h5py.Group's up till root

    :param h5py.Group h5group:
    :param bool includeself:
    :returns generator:
    """
    if includeself:
        yield h5group
    while h5group.parent != h5group:
        h5group = h5group.parent
        yield h5group


def is_link(parent, name):
    """
    Check whether node is h5py.SoftLink or h5py.ExternalLink

    :param h5py.Group parent:
    :param str name:
    :returns bool:
    """
    try:
        lnk = parent.get(name, default=None, getlink=True)
    except (KeyError, RuntimeError):
        return False
    else:
        return isinstance(lnk, (h5py.SoftLink, h5py.ExternalLink))


def h5name(h5group):
    """
    HDF5 Dataset of Group name

    :param h5py.Group h5group:
    :returns str:
    """
    return h5group.name.split('/')[-1]


def nx_class(h5group):
    """
    Nexus class of existing h5py.Group (None when no Nexus instance)

    :param h5py.Group h5group:
    :returns str or None:
    """
    return h5group.attrs.get('NX_class', None)


def is_nx_class(h5group, *classes):
    """
    Nexus class of existing h5py.Group (None when no Nexus instance)

    :param h5py.Group h5group:
    :param \*classes: list(str) of Nexus classes
    :returns bool:
    """
    return nx_class(h5group) in classes


def raise_is_nx_class(h5group, *classes):
    """
    :param h5py.Group h5group:
    :param \*classes: list(str) of Nexus classes
    :raises RuntimeError:
    """
    if is_nx_class(h5group, *classes):
        raise RuntimeError('Nexus class not in {}'.format(classes))


def raise_isnot_nx_class(h5group, *classes):
    """
    :param h5py.Group h5group:
    :param \*classes: list(str) of Nexus classes
    :raises RuntimeError:
    """
    if not is_nx_class(h5group, *classes):
        raise RuntimeError('Nexus class not in {}'.format(classes))


def nx_class_needsinit(parent, name, nxclass):
    """
    Check whether parent[name] needs Nexus initialization

    :param h5py.Group parent:
    :param str or None name:
    :param str nxclass:
    :returns bool: needs initialization
    :raises RuntimeError: wrong Nexus class
    """
    if name is None:
        return nxclass != nx_class(parent)
    if name in parent:
        _nxclass = nx_class(parent[name])
        if _nxclass != nxclass:
            raise RuntimeError('{} is an instance of {} instead of {}'
                               .format(parent[name].name, nxclass, _nxclass))
        return False
    else:
        parent.create_group(name)
        return True


def updated(h5group):
    """
    h5py.Group has changed

    :param h5py.Group h5group:
    """
    tm = timestamp()
    for group in iterup(h5group):
        nxclass = nx_class(group)
        if nxclass is None:
            continue
        elif nxclass in ['NXentry', 'NXsubentry']:
            update_h5dataset(group, 'end_time', tm)
        elif nxclass in ['NXprocess', 'NXnote']:
            update_h5dataset(group, 'date', tm)
        elif nxclass == 'NXroot':
            group.attrs['file_update_time'] = tm


def update_h5dataset(parent, name, data):
    """
    :param h5py.Group parent:
    :param str name:
    :param data:
    """
    try:
        parent[name] = data
    except RuntimeError:
        parent[name][()] = data


def nxclass_init(parent, name, nxclass):
    """
    Initialize Nexus class instance without default attributes and datasets

    :param h5py.Group parent:
    :param str name:
    :raises RuntimeError: wrong Nexus class or parent not an Nexus class instance
    """
    raise_is_nx_class(parent, None)
    if nx_class_needsinit(parent, name, nxclass):
        h5group = parent[name]
        h5group.attrs['NX_class'] = nxclass
        updated(h5group)


def nxroot_init(h5group):
    """
    Initialize NXroot instance

    :param h5py.Group h5group:
    :raises ValueError: not root
    :raises RuntimeError: wrong Nexus class
    """
    if h5group.name != '/':
        raise ValueError('Group should be the root')
    if nx_class_needsinit(h5group, None, 'NXroot'):
        h5group.attrs['file_time'] = timestamp()
        h5group.attrs['file_name'] = vlen_string(h5group.file.filename)
        h5group.attrs['HDF5_Version'] = vlen_string(h5py.version.hdf5_version)
        h5group.attrs['h5py_version'] = vlen_string(h5py.version.version)
        h5group.attrs['creator'] = PROGRAM_NAME
        h5group.attrs['NX_class'] = 'NXroot'
        updated(h5group)


def nxentry_init(parent, name):
    """
    Initialize NXentry instance

    :param h5py.Group parent:
    :raises RuntimeError: wrong Nexus class or parent not NXroot
    """
    raise_isnot_nx_class(parent, 'NXroot')
    if nx_class_needsinit(parent, name, 'NXentry'):
        h5group = parent[name]
        update_h5dataset(h5group, 'start_time', timestamp())
        h5group.attrs['NX_class'] = 'NXentry'
        updated(h5group)


def nxnote_init(parent, name, data, type):
    """
    Initialize NXnote instance

    :param h5py.Group parent:
    :param str name:
    :param str data:
    :param str type:
    :raises RuntimeError: wrong Nexus class or parent not an Nexus class instance
    """
    raise_is_nx_class(parent, None)
    if nx_class_needsinit(parent, name, 'NXnote'):
        h5group = parent[name]
        update_h5dataset(h5group, 'data', vlen_string(data))
        update_h5dataset(h5group, 'type', vlen_string(type))
        h5group.attrs['NX_class'] = 'NXnote'
        updated(h5group)


def nxprocess_init(parent, name, configdict=None):
    """
    Initialize NXprocess instance

    :param h5py.Group parent:
    :param str name:
    :raises RuntimeError: wrong Nexus class or parent not NXentry
    """
    raise_isnot_nx_class(parent, 'NXentry')
    if nx_class_needsinit(parent, name, 'NXprocess'):
        h5group = parent[name]
        update_h5dataset(h5group, 'program', PROGRAM_NAME)
        update_h5dataset(h5group, 'version', PROGRAM_VERSION)
        h5group.attrs['NX_class'] = 'NXprocess'
        updated(h5group)
    else:
        h5group = parent[name]
    if configdict is not None:
        data = configdict.tostring()
        nxnote_init(h5group, 'configuration', data, 'ini')
    nxclass_init(h5group, 'results', 'NXcollection')


@contextmanager
def nxroot(path, mode='r', **kwargs):
    """
    h5py.File context with NXroot initialization

    :param str path:
    :param str mode: h5py.File modes
    :param \**kwargs: see h5py.File
    :returns contextmanager:
    """
    if mode != 'r':
        mkdir(os.path.dirname(path))
    with h5py.File(path, mode=mode, **kwargs) as h5file:
        nxroot_init(h5file)
        yield h5file


def nxentry(root, name):
    """
    Get NXentry instance (initialize when missing)

    :param h5py.Group root:
    :param str name:
    :returns h5py.Group:
    """
    nxentry_init(root, name)
    return root[name]


def nxprocess(entry, name, **kwargs):
    """
    Get NXprocess instance (initialize when missing)

    :param h5py.Group entry:
    :param str name:
    :param \**kwargs: see nxprocess_init
    :returns h5py.Group:
    """
    nxprocess_init(entry, name, **kwargs)
    return entry[name]


def nxcollection(parent, name):
    """
    Get NXcollection instance (initialize when missing)

    :param h5py.Group parent:
    :param str name:
    :returns h5py.Group:
    """
    nxclass_init(parent, name, 'NXcollection')
    return parent[name]


def nxdata(parent, name):
    """
    Get NXcollection instance (initialize when missing)

    :param h5py.Group parent:
    :param str or None name:
    :returns h5py.Group:
    """
    if name is None:
        name = DEFAULT_PLOT_NAME
    nxclass_init(parent, name, 'NXdata')
    return parent[name]


def nxdata_add_axes(data, axes, append=True):
    """
    Add axes to NXdata instance

    :param h5py.Group data:
    :param list(3-tuple) axes: name(str), value(None,h5py.Dataset,numpy.ndarray), attrs(dict)
    :param bool append:
    """
    raise_isnot_nx_class(data, 'NXdata')
    if append:
        newaxes = data.attrs.get('axes', [])
    else:
        newaxes = []
    for name, value, attrs in axes:
        if value is None:
            pass  # is or will be created elsewhere
        elif isinstance(value, h5py.Dataset):
            if value.parent != data:
                data[name] = h5py.SoftLink(value.name)
        elif isinstance(value, dict):
            data.create_dataset(name, **value)
        else:
            data[name] = value
        if attrs:
            data[name].attrs.update(attrs)
        newaxes.append(name)
    if newaxes:
        data.attrs['axes'] = vlen_string(newaxes)
        updated(data)


def nxdata_get_signals(data):
    """
    Get NXdata signals (default signal first)

    :param h5py.Group data:
    :returns list(str): signal names (default first)
    """
    signal = data.attrs.get('signal', None)
    auxsignals = data.attrs.get('auxiliary_signals', None)
    if signal is None:
        if auxsignals is None:
            return []
        else:
            return auxsignals.tolist()
    else:
        return [signal] + auxsignals.tolist()


def nxdata_set_signals(data, signals):
    """
    Set NXdata signals (default signal first)

    :param h5py.Group data:
    :param list(str) signals:
    """
    if signals:
        data.attrs['signal'] = vlen_string(signals[0])
        if len(signals)>1:
            data.attrs['auxiliary_signals'] = vlen_string(signals[1:])
        else:
            data.attrs.pop('auxiliary_signals', None)
    else:
        data.attrs.pop('signal', None)
        data.attrs.pop('auxiliary_signals', None)
    updated(data)


def nxdata_add_signals(data, signals, append=True):
    """
    Add signals to NXdata instance

    :param h5py.Group data:
    :param list(2-tuple) signals: name(str), value(None,h5py.Dataset,numpy.ndarray,dict), attrs(dict)
    :param bool append:
    """
    raise_isnot_nx_class(data, 'NXdata')
    if append:
        newsignals = nxdata_get_signals(data)
    else:
        newsignals = []
    for name, value, attrs in signals:
        if value is None:
            pass  # is or will be created elsewhere
        elif isinstance(value, h5py.Dataset):
            if value.parent != data:
                data[name] = h5py.SoftLink(value.name)
        elif isinstance(value, dict):
            data.create_dataset(name, **value)
        else:
            data[name] = value
        if attrs:
            data[name].attrs.update(attrs)
        newsignals.append(name)
    if newsignals:
        nxdata_set_signals(data, newsignals)


def mark_default(h5group):
    """
    Mark HDF5 Dataset or Group as default (parents get notified as well)

    :param h5py.Group h5group:
    """
    path = h5group
    nxclass = nx_class(path)
    nxdata = None
    for parent in iterup(path.parent):
        parentnxclass = nx_class(parent)
        if parentnxclass == 'NXdata':
            signals = nxdata_get_signals(parent)
            signal = h5name(path)
            if signal in signals:
                signals.pop(signals.index(signal))
            nxdata_set_signals(parent, [signal]+signals)
            updated(parent)
        elif nxclass == 'NXentry':
            parent.attrs['default'] = h5name(path)
            updated(parent)
        elif parentnxclass is not None:
            if nxclass == 'NXdata':
                nxdata = path
            else:
                if nxdata:
                    if is_link(parent, DEFAULT_PLOT_NAME):
                        del parent[DEFAULT_PLOT_NAME]
                    parent[DEFAULT_PLOT_NAME] = h5py.SoftLink(nxdata.name)
                    nxdata = parent[DEFAULT_PLOT_NAME]
            if nxdata:
                parent.attrs['default'] = h5name(nxdata)
                updated(parent)
            if parentnxclass == 'NXroot':
                break
        path = parent
        nxclass = parentnxclass

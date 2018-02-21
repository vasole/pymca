#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from operator import itemgetter
import re
import posixpath
from h5py import Dataset, Group
DEBUG = 0

#sorting method
def h5py_sorting(object_list):
    sorting_list = ['start_time', 'end_time', 'name']
    n = len(object_list)
    if n < 2:
        return object_list

    # we have received items, not values
    # perform a first sort based on received names
    # this solves a problem with Eiger data where all the
    # external data have the same posixName. Without this sorting
    # they arrive "unsorted"
    object_list.sort()
    try:
        posixNames = [item[1].name for item in object_list]
    except AttributeError:
        # Typical of broken external links
        if DEBUG:
            print("HDF5Widget: Cannot get posixNames")
        return object_list

    # This implementation only sorts entries
    if posixpath.dirname(posixNames[0]) != "/":
        return object_list

    sorting_key = None
    if hasattr(object_list[0][1], "items"):
        for key in sorting_list:
            if key in [x[0] for x in object_list[0][1].items()]:
                sorting_key = key
                break

    if sorting_key is None:
        if 'name' in sorting_list:
            sorting_key = 'name'
        else:
            return object_list

    try:
        if sorting_key != 'name':
            sorting_list = [(o[1][sorting_key].value, o)
                           for o in object_list]
            sorted_list = sorted(sorting_list, key=itemgetter(0))
            return [x[1] for x in sorted_list]

        if sorting_key == 'name':
            sorting_list = [(_get_number_list(o[1].name),o)
                           for o in object_list]
            sorting_list.sort()
            return [x[1] for x in sorting_list]
    except:
        #The only way to reach this point is to have different
        #structures among the different entries. In that case
        #defaults to the unfiltered case
        print("WARNING: Default ordering")
        print("Probably all entries do not have the key %s" % sorting_key)
        return object_list

def _get_number_list(txt):
    rexpr = '[/a-zA-Z:-]'
    nbs= [float(w) for w in re.split(rexpr, txt) if w not in ['',' ']]
    return nbs

def isGroup(item):
    if isinstance(item, Group):
        return True
    elif hasattr(item, "keys"):
        return True
    else:
        return False

def isDataset(item):
    if isinstance(item, Dataset):
        return True
    else:
        return False

def getEntryName(path):
    """
    Retrieve the top level name (not h5py object) associated to a given path
    despite being or not an NXentry group.
    """
    entry = path
    candidate = posixpath.dirname(entry)
    while len(candidate) > 1:
        entry = candidate
        candidate = posixpath.dirname(entry)
    return entry

def getNXClassGroups(h5file, path, classes, single=False):
    """
    Retrieve the hdf5 groups inside a given path where the NX_class attribute
    matches one of the items in the classes list.
    """
    groups = []
    items_list = list(h5file[path].items())
    if ("NXentry" in classes) or (b"NXentry" in classes):
        items_list = h5py_sorting(items_list)
    for key, group in items_list:
        if not isGroup(group):
            continue
        for attr in group.attrs:
            if attr in ["NX_class", b"NX_class"]:
                if group.attrs[attr] in classes:
                    groups.append(group)
                    if single:
                        break
    return groups

def getPositionersGroup(h5file, path):
    """
    Retrieve the positioners group associated to a path
    retrieving them from the same entry (assuming they are in
    NXentry/instrument/positioners)
    """
    entry_path = getEntryName(path)
    instrument = getNXClassGroups(h5file, entry_path, ["NXinstrument", b"NXinstrument"], single=True)
    positioners = None
    if len(instrument):
        instrument = instrument[0]
        for key in instrument.keys():
            if key in ["positioners", b"positioners"]:
                positioners = instrument[key]
                if not isGroup(positioners):
                    positioners = None
    if positioners is None:
        # sardana stores the positioners inside measurement/pre_scan_snapshot
        entry = h5file[entry_path]
        sardana = "measurement/pre_scan_snapshot"
        if sardana in entry:
            group = entry[sardana]
            if isGroup(group):
                positioners = group
    return positioners

def getMeasurementGroup(h5file, path):
    if path in ["/", b"/", "", b""]:
        raise ValueError("path cannot be the toplevel root")
    entry_path = getEntryName(path)
    entry = h5file[entry_path]
    items_list = entry.items()
    measurement = None
    for key, group in items_list:
        if key in ["measurement", b"measurement"]:
            if isGroup(group):
                measurement = group
    if measurement is None:
        # try to get the default NXdata groups as measurement group
        default = None
        for attr in entry.attrs:
            if attr in ["default", b"default"]:
                default = entry.attrs[attr]
        # hdf5 stores in utf-8 the paths if we got bytes, they need to be converted
        if hasattr(default, "decode"):
            default = default.decode()
        if default is None:            
            # get the NXdata group just behind entry that contains more items inside
            # and take it as measurement group
            nxdatas = getNXClassGroups(h5file, entry_path, ["NXdata", b"NXdata"], single=False)
            if len(nxdatas):
                measurement = nxdatas[0]
                nitems = len(measurement)
            for group in nxdatas:
                if len(group) > nitems:
                    measurement = group
                    nitems = len(measurement)
        else:
            # default could ne anything ... crashes should be prevented
            if default in entry:
                group = entry[default]
                if isGroup(group):
                    measurement = group
    return measurement
    
def getInstrumentGroup(h5file, path):
    entry_name = getEntryName(path)
    groups = getNXClassGroups(h5file, entry_name, ["NXinstrument", b"NXinstrument"] , single=False)
    n = len(groups)
    if n == 0:
        return None
    else:
        if n > 1:
            print("WARNING: More than one instrument associated to the same entry")
        return groups[0]

def getScannedPositioners(h5file, path):
    """
    Try to retrieve the positioners (aka. motors) that were moved
    For that:
        - Look for datasets present at measurement and positioners groups
        - Look for positioners with more than one single value
        - Look for datasets present at measurement and title
    """
    entry_name = getEntryName(path)
    measurement = getMeasurementGroup(h5file, entry_name)
    scanned = []
    if measurement is not None:
        positioners = getPositionersGroup(h5file, entry_name)
        if positioners is not None:
            priorityPositioners = False
            if priorityPositioners:
                counters = [key for key, item in measurement.items() if isDataset(item)]
                scanned = [item.name for key, item in positioners.items() if key in counters]
            else:
                motors = [key for key, item in positioners.items() if isDataset(item)]
                scanned = [item.name for key, item in measurement.items() if key in motors]
                if len(scanned) > 1:
                    # check that motors are not duplicated without reason
                    scanned = [item.name for key, item in measurement.items() if \
                                          (key in motors) and \
                                          (hasattr(item, "size") and (item.size > 1))]
            if not len(scanned):
                # look for datasets with more than one single value inside positioners
                scanned = [item.name for key, item in positioners.items() if \
                                            isDataset(item) and \
                                            (hasattr(item, "size") and (item.size > 1))]
        if not len(scanned):
            entry = h5file[entry_name]
            if "title" in entry:
                title = entry["title"].value
                if hasattr(title, "decode"):
                    title = title.decode("utf-8")
                tokens = title.split()
                candidates = [key for key, item in measurement.items() if \
                                            isDataset(item) and \
                                            (key in tokens)]
                indices = []
                for key in candidates:
                    indices.append((tokens.index(key), key))
                indices.sort()
                if len(indices):
                    scanned = [measurement[key].name for idx, key in indices]        
    return scanned

if __name__ == "__main__":
    import sys
    import h5py
    try:
        sourcename=sys.argv[1]
    except:
        print("Usage: NexusTools <file> <key>")
        sys.exit()
    h5 = h5py.File(sourcename)    
    entries = getNXClassGroups(h5, "/", ["NXentry", b"NXentry"], single=False)
    for entry in entries:
        print("Entry name = %s" % entry.name)
        if "title" in entry:
            print("Entry title = %s" % entry["title"].value)
        measurement = getMeasurementGroup(h5, entry.name)
        if measurement is None:
            print("No measurement")
        else:
            print("Measurement name = %s " % measurement.name)
        instrument = getInstrumentGroup(h5, entry.name)
        if instrument is None:
            print("No instrument")
        else:
            print("Instrument name = %s " % instrument.name)
        positioners = getPositionersGroup(h5, entry.name)
        if positioners is None:
            print("No positioners")
        else:
            print("Positioners name = %s " % positioners.name)
        scanned = getScannedPositioners(h5, entry.name)
        if len(scanned):
            for i in range(len(scanned)):
                print("Scanned motors %d = %s" % (i, scanned[i]))
        else:
            print("Unknown scanned motors")

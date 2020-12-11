#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2020 European Synchrotron Radiation Facility
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
import sys
import os
import numpy
try:
    # try to import hdf5plugin
    import hdf5plugin
except:
    # but do not crash just because of it
    pass
import h5py
from operator import itemgetter
import re
import posixpath
import logging
phynx = h5py

if sys.version_info >= (3,):
    basestring = str

from . import DataObject
from . import NexusTools

SOURCE_TYPE = "HDF5"

try:
    from silx.io import open as silxh5open
    logging.getLogger("silx.io.fabioh5").setLevel(logging.CRITICAL)
except:
    silxh5open = None

def h5open(filename):
    try:
        # try to open as usual using h5py
        return h5py.File(filename, "r")
    except:
        try:
            if h5py.version.hdf5_version_tuple < (1, 10):
                # no reason to try SWMR mode
                raise
            elif h5py.is_hdf5(filename):
                _logger.info("Cannot open %s. Trying in SWMR mode" % filename)
                return h5py.File(filename, "r", libver='latest', swmr=True)
            else:
                raise
        except:
            if silxh5open:
                try:
                    _logger.info("Trying to open %s using silx" % filename)
                    return silxh5open(filename)
                except:
                    _logger.info("Cannot open %s using silx" % filename)
            # give back the original error
            return h5py.File(filename, "r")

_logger = logging.getLogger(__name__)

#sorting method
def h5py_sorting(object_list):
    sorting_list = ['start_time', 'end_time', 'name']
    n = len(object_list)
    if n < 2:
        return object_list

    # This implementation only sorts entries
    if posixpath.dirname(object_list[0].name) != "/":
        return object_list

    names = list(object_list[0].keys())

    sorting_key = None
    for key in sorting_list:
        if key in names:
            sorting_key = key
            break

    if sorting_key is None:
        if 'name' in sorting_list:
            sorting_key = 'name'
        else:
            return object_list

    try:
        if sorting_key != 'name':
            sorting_list = [(o[sorting_key][()], o)
                           for o in object_list]
            sorted_list = sorted(sorting_list, key=itemgetter(0))
            return [x[1] for x in sorted_list]

        if sorting_key == 'name':
            sorting_list = [(_get_number_list(o.name),o)
                           for o in object_list]
            sorting_list.sort()
            return [x[1] for x in sorting_list]
    except:
        #The only way to reach this point is to have different
        #structures among the different entries. In that case
        #defaults to the unfiltered case
        _logger.warning("Default ordering")
        _logger.warning("Probably all entries do not have the key %s", sorting_key)
        return object_list

def _get_number_list(txt):
    rexpr = '[/a-zA-Z:-]'
    nbs= [float(w) for w in re.split(rexpr, txt) if w not in ['',' ']]
    return nbs

def get_family_pattern(filelist):
    name1 = filelist[0]
    name2 = filelist[1]
    if name1 == name2:
        return name1
    i0=0
    for i in range(len(name1)):
        if i >= len(name2):
            break
        elif name1[i] == name2[i]:
            pass
        else:
            break
    i0 = i
    for i in range(i0,len(name1)):
        if i >= len(name2):
            break
        elif name1[i] != name2[i]:
            pass
        else:
            break
    i1 = i
    if i1 > 0:
        delta=1
        while (i1-delta):
            if (name2[(i1-delta)] in ['0', '1', '2',
                                    '3', '4', '5',
                                    '6', '7', '8',
                                    '9']):
                delta = delta + 1
            else:
                if delta > 1: delta = delta -1
                break
        fmt = '%dd' % delta
        if delta > 1:
            fmt = "%0" + fmt
        else:
            fmt = "%" + fmt
        rootname = name1[0:(i1-delta)]+fmt+name2[i1:]
    else:
        rootname = name1[0:]
    return rootname

class NexusDataSource(object):
    def __init__(self,nameInput):
        if type(nameInput) == type([]):
            nameList = nameInput
        else:
            nameList = [nameInput]
        self.sourceName = []
        for name in nameList:
            if not isinstance(name, basestring):
                if not isinstance(name, h5py.File):
                    text = "Constructor needs string as first argument"
                    raise TypeError(text)
                else:
                    self.sourceName.append(name.file)
                    continue
            self.sourceName.append(name)
        self.sourceType = SOURCE_TYPE
        self.__sourceNameList = self.sourceName
        self._sourceObjectList=[]
        self.refresh()

    def refresh(self):
        for instance in self._sourceObjectList:
            instance.close()
        self._sourceObjectList=[]
        FAMILY = False
        for name in self.__sourceNameList:
            if isinstance(name, h5py.File):
                self._sourceObjectList.append(name)
                continue
            if not os.path.exists(name):
                if '%' in name:
                   phynxInstance = h5py.File(name, 'r',
                                              driver='family')
                else:
                    raise IOError("File %s does not exists" % name)
            try:
                phynxInstance = h5open(name)
            except IOError:
                if 'FAMILY DRIVER' in sys.exc_info()[1].args[0].upper():
                    FAMILY = True
                else:
                    raise
            except TypeError:
                try:
                    phynxInstance = h5open(name)
                except IOError:
                    if 'FAMILY DRIVER' in sys.exc_info()[1].args[0].upper():
                        FAMILY = True
                    else:
                        raise
            if FAMILY and (len(self._sourceObjectList) > 0):
                txt = "Mixing segmented and non-segmented HDF5 files not supported yet"
                raise IOError(txt)
            elif FAMILY:
                break
            phynxInstance._sourceName = name
            self._sourceObjectList.append(phynxInstance)
        if FAMILY:
            pattern = get_family_pattern(self.__sourceNameList)
            if '%' in pattern:
                phynxInstance = h5py.File(pattern, 'r',
                                            driver='family')
            else:
                raise IOError("Cannot read set of HDF5 files")
            self.sourceName   = [pattern]
            self.__sourceNameList = [pattern]
            self._sourceObjectList=[phynxInstance]
            phynxInstance._sourceName = pattern
        self.__lastKeyInfo = {}

    def getSourceInfo(self):
        """
        Returns a dictionary with the key "KeyList" (list of all available keys
        in this source). Each element in "KeyList" has the form 'n1.n2' where
        n1 is the source number and n2 entry number in file both starting at 1.
        """
        return self.__getSourceInfo()


    def __getSourceInfo(self):
        SourceInfo={}
        SourceInfo["SourceType"]=SOURCE_TYPE
        SourceInfo["KeyList"]=[]
        i = 0
        for sourceObject in self._sourceObjectList:
            i+=1
            nEntries = len(sourceObject["/"].keys())
            for n in range(nEntries):
                SourceInfo["KeyList"].append("%d.%d" % (i,n+1))
        SourceInfo["Size"]=len(SourceInfo["KeyList"])
        return SourceInfo

    def getKeyInfo(self, key):
        if key in self.getSourceInfo()['KeyList']:
            return self.__getKeyInfo(key)
        else:
            #should we raise a KeyError?
            _logger.debug("Error key not in list ")
            return {}

    def __getKeyInfo(self,key):
        try:
            index, entry = key.split(".")
            index = int(index)-1
            entry = int(entry)-1
        except:
            #should we rise an error?
            _logger.debug("Error trying to interpret key = %s", key)
            return {}

        sourceObject = self._sourceObjectList[index]
        info = {}
        info["SourceType"]  = SOURCE_TYPE
        #doubts about if refer to the list or to the individual file
        info["SourceName"]  = self.sourceName[index]
        info["Key"]         = key
        #specific info of interest
        info['FileName'] = sourceObject.name
        return info

    def getDataObject(self, key, selection=None):
        """
        key:  a string of the form %d.%d indicating the file and the entry
              starting by 1.
        selection: a dictionary generated via QNexusWidget
        """
        _logger.debug("getDataObject selection = %s", selection)
        if selection is not None:
            if 'sourcename' in selection:
                filename  = selection['sourcename']
                entry     = selection['entry']
                fileIndex  = self.__sourceNameList.index(filename)
                phynxFile =  self._sourceObjectList[fileIndex]
                if entry == "/":
                    entryIndex = 0
                else:
                    entryIndex = list(phynxFile["/"].keys()).index(entry[1:])
            else:
                key_split = key.split(".")
                fileIndex = int(key_split[0])-1
                phynxFile =  self._sourceObjectList[fileIndex]
                entryIndex = int(key_split[1])-1
                entry = phynxFile["/"].keys()[entryIndex]
            actual_key = "%d.%d" % (fileIndex+1, entryIndex+1)
            if actual_key != key:
                if entry != "/":
                    _logger.warning("selection keys do not match")
        else:
            #Probably I should find the acual entry following h5py_ordering output
            #and search for an NXdata plot.
            sourcekeys = self.getSourceInfo()['KeyList']
            #a key corresponds to an image
            key_split= key.split(".")
            actual_key= "%d.%d" % (int(key_split[0]), int(key_split[1]))
            if actual_key not in sourcekeys:
                raise KeyError("Key %s not in source keys" % actual_key)
            raise NotImplemented("Direct NXdata plot not implemented yet")
        #create data object
        output = DataObject.DataObject()
        output.x = None
        output.y = None
        output.m = None
        output.data = None
        output.info = self.__getKeyInfo(actual_key)
        try:
            output.info["title"] = NexusTools.getTitle(phynxFile, entry)
        except:
            txt = "Error reading title for path <%s>"
            _logger.warning(txt)
            output.info["title"] = ""
        output.info['selection'] = selection
        if "mca" in selection:
            # this should go somewhere else
            h5File = phynxFile
            mcaPath = entry + selection["mcalist"][selection["mca"][0]]
            mcaObjectPaths = NexusTools.getMcaObjectPaths(phynxFile, mcaPath)
            mcaData = h5File[mcaObjectPaths['counts']]
            output.info['selectiontype'] = "1D"
            try:
                for key in list(mcaObjectPaths.keys()):
                    if key == "counts":
                        continue
                    mcaDatasetObjectPath = mcaObjectPaths[key]
                    dataset = None
                    if mcaDatasetObjectPath in h5File:
                        dataset = h5File[mcaDatasetObjectPath][()]
                    elif "::" in mcaDatasetObjectPath:
                        fileName, path = mcaDatasetObjectPath.split()
                        if os.path.exists(fileName):
                            with h5open(fileName) as h5:
                                if path in h5:
                                    dataset = h5[path][()]
                    if dataset is None:
                        _logger.debug("Broken link? Ignoring key %s = %s",
                                      key, mcaDatasetObjectPath)
                        del mcaObjectPaths[key]
                    else:
                        mcaObjectPaths[key] = dataset
                if "channels" in mcaObjectPaths:
                    mcaChannels = mcaObjectPaths["channels"]
                    del mcaObjectPaths["channels"]
                else:
                    mcaChannels = numpy.arange(mcaData.shape[-1]).astype(numpy.float32)
                if "calibration" in mcaObjectPaths:
                    mcaCalibration = mcaObjectPaths["calibration"]
                    del mcaObjectPaths["calibration"]
                else:
                    mcaCalibration = numpy.array([0.0, 1.0, 0.0])
                output.info["McaCalib"] = mcaCalibration
                if "preset_time" in mcaObjectPaths:
                    output.info["McaPresetTime"]= mcaObjectPaths["preset_time"]
                    del mcaObjectPaths["preset_time"]
                if "elapsed_time" in mcaObjectPaths:
                    output.info["McaRealTime"]= mcaObjectPaths["elapsed_time"]
                    del mcaObjectPaths["elapsed_time"]
                if "live_time" in mcaObjectPaths:
                    output.info["McaLiveTime"]= mcaObjectPaths["live_time"]
                    del mcaObjectPaths["live_time"]
                del mcaObjectPaths
                if selection['mcaselectiontype'].lower() in ["avg", "average", "sum"]:
                    divider = 1.0
                    if len(mcaData.shape) > 1:
                        divider *= mcaData.shape[0]
                        mcaData = numpy.sum(mcaData, axis=0, dtype=numpy.float32)
                        while len(mcaData.shape) > 1:
                            divider *= mcaData.shape[0]
                            mcaData = mcaData.sum(axis=0)
                        if selection['mcaselectiontype'].lower() != "sum":
                            mcaData /= divider
                    else:
                        mcaData = mcaData[()]
                        divider = 1.0
                    if "McaLiveTime" in output.info:
                        if numpy.isscalar(output.info["McaLiveTime"]):
                            # it is already a single number
                            pass
                        else:
                            output.info["McaLiveTime"] = \
                                    output.info["McaLiveTime"].sum()
                        if selection['mcaselectiontype'].lower() != "sum":
                            output.info["McaLiveTime"] /= divider
            except:
                # import traceback
                _logger.error("%s", sys.exc_info())
                # print(("%s " % value) + ''.join(traceback.format_tb(trace)))
                return output
            output.x = [mcaChannels]
            output.y = [mcaData]
            return output
        elif selection['selectiontype'].upper() in ["SCAN", "MCA"]:
            output.info['selectiontype'] = "1D"
        elif selection['selectiontype'] == "3D":
            output.info['selectiontype'] = "3D"
        elif selection['selectiontype'] == "2D":
            output.info['selectiontype'] = "2D"
            output.info['imageselection'] = True
        else:
            raise TypeError("Unsupported selection type %s" %\
                            selection['selectiontype'])
        if 'LabelNames' in selection:
            output.info['LabelNames'] = selection['LabelNames']
        elif 'aliaslist' in selection:
            output.info['LabelNames'] = selection['aliaslist']
        else:
            output.info['LabelNames'] = selection['cntlist']
        if entry != "/":
            try:
                positioners = NexusTools.getPositionersGroup(phynxFile, entry)
                if positioners is not None:
                    output.info['MotorNames'] = []
                    output.info['MotorValues'] = []
                    for key in positioners.keys():
                        if positioners[key].dtype in [object, numpy.object_]:
                            # not a standard value
                            _logger.info("Skipping object key %s" % key)
                            continue
                        output.info['MotorNames'].append(key)
                        value = positioners[key][()]
                        if hasattr(value, "size"):
                            if value.size > 1:
                                if hasattr(value, "flat"):
                                    value = value.flat[0]
                        output.info['MotorValues'].append(value)
            except:
                # I cannot affort to fail here for something probably not used
                _logger.debug("Error reading positioners\n%s", sys.exc_info())
        for cnt in ['y', 'x', 'm']:
            if not cnt in selection:
                continue
            if not len(selection[cnt]):
                continue
            path =  entry + selection['cntlist'][selection[cnt][0]]
            data = phynxFile[path]
            totalElements = 1
            for dim in data.shape:
                totalElements *= dim
            if totalElements < 2.0E7:
                try:
                    data = phynxFile[path][()]
                except MemoryError:
                    data = phynxFile[path]
                    pass
            if output.info['selectiontype'] in ["1D", "MCA"]:
                if (len(data.shape) > 1) and ('mcaselectiontype' in selection):
                    mcaselectiontype = selection['mcaselectiontype'].lower()
                    nSpectra = 1.0
                    for iDummy in data.shape[:-1]:
                        data = data.sum(axis=0, dtype=numpy.float64)
                        nSpectra *= iDummy
                    if mcaselectiontype == "sum":
                        # sum already calculated
                        _logger.debug("SUM")
                    elif mcaselectiontype in ["avg", "average"]:
                        # calculate the average
                        _logger.debug("AVERAGE")
                        data /= nSpectra
                    else:
                        _logger.warning("Unsupported selection type %s",
                                        mcaselectiontype)
                        _logger.warning("Calculating average")
                        data /= nSpectra
                elif len(data.shape) == 2:
                    if min(data.shape) == 1:
                        data = numpy.ravel(data)
                    else:
                        raise TypeError("%s selection is not 1D" % cnt.upper())
                elif len(data.shape) > 2:
                    raise TypeError("%s selection is not 1D" % cnt.upper())
            if cnt == 'y':
                if output.info['selectiontype'] == "2D":
                    output.data = data
                else:
                    output.y = [data]
            elif cnt == 'x':
                #there can be more than one X except for 1D
                if output.info['selectiontype'] == "1D":
                    if len(selection[cnt]) > 1:
                        raise TypeError("%s selection is not 1D" % cnt.upper())
                if output.x is None:
                    output.x = [data]
                if len(selection[cnt]) > 1:
                    for xidx in range(1, len(selection[cnt])):
                        path =  entry + selection['cntlist'][selection[cnt][xidx]]
                        data = phynxFile[path][()]
                        output.x.append(data)
            elif cnt == 'm':
                #only one monitor
                output.m = [data]

        # SCAN specific to handle asynchronous writing
        if output.info['selectiontype'] in ["1D"]:
            if output.y:
                length = ylength = output.y[0].size
                delta = 0
                if output.x:
                    xlength = output.x[0].size
                    delta = abs(ylength - xlength)
                    length = min(length, xlength)
                if output.m:
                    mlength = output.m[0].size
                    delta = max(delta, ylength - mlength)
                    length = min(length, mlength)
                if delta > 1:
                    _logger.warning("Stripping last %d points" % delta)
                elif delta == 1:
                    _logger.info("Stripping last point of selection")
                if delta > 0:
                    for i in range(len(output.y)):
                        output.y[i] = output.y[i][:length]
                    if output.x:
                        for i in range(len(output.x)):
                            output.x[i] = output.x[i][:length]
                    if output.m:
                        for i in range(len(output.m)):
                            output.m[i] = output.m[i][:length]

        # MCA specific
        if selection['selectiontype'].upper() == "MCA":
            if not 'Channel0' in output.info:
                output.info['Channel0'] = 0
        """"
        elif selection['selectiontype'].upper() in ["BATCH"]:
            #assume already digested
            output.x = None
            output.y = None
            output.m = None
            output.data = None
            entryGroup = phynxFile[entry]
            output.info['Channel0'] = 0
            for key in ['y', 'x', 'm', 'data']:
                if key not in selection:
                    continue
                if type(selection[key]) != type([]):
                    selection[key] = [selection[key]]
                if not len(selection[key]):
                    continue
                for cnt in selection[key]:
                    dataset = entryGroup[cnt]
                    if cnt == 'y':
                        if output.y is None:
                            output.y = [dataset]
                        else:
                            output.y.append(dataset)
                    elif cnt == 'x':
                        if output.x is None:
                            output.x = [dataset]
                        else:
                            output.x.append(dataset)
                    elif cnt == 'm':
                        if output.m is None:
                            output.m = [dataset]
                        else:
                            output.m.append(dataset)
                    elif cnt == 'data':
                        if output.data is None:
                            output.data = [dataset]
                        else:
                            output.data.append(dataset)
        """
        return output

    def isUpdated(self, sourceName, key):
        #sourceName is redundant?
        index, entry = key.split(".")
        index = int(index)-1
        lastmodified = os.path.getmtime(self.__sourceNameList[index])
        if lastmodified != self.__lastKeyInfo[key]:
            self.__lastKeyInfo[key] = lastmodified
            return True
        else:
            return False

source_types = { SOURCE_TYPE: NexusDataSource}

def DataSource(name="", source_type=SOURCE_TYPE):
  try:
     sourceClass = source_types[source_type]
  except KeyError:
     #ERROR invalid source type
     raise TypeError("Invalid Source Type, source type should be one of %s" %\
                     source_types.keys())
  return sourceClass(name)


if __name__ == "__main__":
    import time
    try:
        sourcename=sys.argv[1]
        key       =sys.argv[2]
    except:
        print("Usage: NexusDataSource <file> <key>")
        sys.exit()
    #one can use this:
    obj = NexusDataSource(sourcename)
    #or this:
    obj = DataSource(sourcename)
    #data = obj.getData(key,selection={'pos':(10,10),'size':(40,40)})
    #data = obj.getDataObject(key,selection={'pos':None,'size':None})
    t0 = time.time()
    data = obj.getDataObject(key,selection=None)
    print("elapsed = ",time.time() - t0)
    print("info = ",data.info)
    if data.data is not None:
        print("data shape = ",data.data.shape)
        print(numpy.ravel(data.data)[0:10])
    else:
        print(data.y[0].shape)
        print(numpy.ravel(data.y[0])[0:10])
    data = obj.getDataObject('1.1',selection=None)
    r = int(key.split('.')[-1])
    print(" data[%d,0:10] = " % (r-1),data.data[r-1   ,0:10])
    print(" data[0:10,%d] = " % (r-1),data.data[0:10, r-1])

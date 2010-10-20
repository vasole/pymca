#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
__revision__ = "$Revision: 1.1 $"
import numpy
import DataObject
import sys
import os
import types
import h5py
import re
import posixpath

try:
    import PyMca.phynx as phynx
except ImportError:
    try:
        import phynx
    except ImportError:
        from xpaxs.io import phynx

SOURCE_TYPE = "HDF5"
DEBUG = 0

def h5py_sorting(object_list):
    sorting_list = ['start_time', 'end_time', 'name']
    n = len(object_list)
    if n < 2:
        return object_list

    # This implementation only sorts entries
    if posixpath.dirname(object_list[0].name) != "/":
        return object_list

    names = object_list[0].keys()

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
            sorting_list = [(o[sorting_key].value, o)
                           for o in object_list]
            sorting_list.sort()
            return [x[1] for x in sorting_list]

        if sorting_key == 'name':
            sorting_list = [(_get_number_list(o.name),o)
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


class NexusDataSource:
    def __init__(self,nameInput):
        if type(nameInput) == types.ListType:
            nameList = nameInput
        else:
            nameList = [nameInput]
        self.sourceName = []
        for name in nameList:
            if type(name) != types.StringType:
                if not isinstance(name, phynx.File):
                    text = "Constructor needs string as first argument"
                    raise TypeError(text)
                else:
                    self.sourceName.append(name.file)
                    continue
            self.sourceName.append(name)
        self.sourceType = SOURCE_TYPE
        self.__sourceNameList = self.sourceName
        self.refresh()
        
    def refresh(self):
        self._sourceObjectList=[]
        FAMILY = False
        for name in self.__sourceNameList:
            if isinstance(name, phynx.File):
                self._sourceObjectList.append(name)
                continue
            if not os.path.exists(name):
                if '%' in name:
                    try:
                        phynxInstance = phynx.File(name, 'r',
                           driver='family',
                           lock=None,
                           sorted_with=h5py_sorting)
                    except TypeError:
                        phynxInstance = phynx.File(name, 'r',
                                               driver='family',
                                               lock=None)
                else:
                    raise IOError, "File %s does not exists" % name
            try:
                phynxInstance = phynx.File(name, 'r', lock=None,
                                           sorted_with=h5py_sorting)
            except IOError:
                if 'FAMILY DRIVER' in sys.exc_info()[1].args[0].upper():
                    FAMILY = True
                else:
                    raise
            except TypeError:
                try:
                    phynxInstance = phynx.File(name, 'r', lock=None)
                except IOError:
                    if 'FAMILY DRIVER' in sys.exc_info()[1].args[0].upper():
                        FAMILY = True
                    else:
                        raise
            if FAMILY and (len(self._sourceObjectList) > 0):
                raise IOError, "Mixing segmented and non-segmented HDF5 files not supported yet"
            elif FAMILY:
                break
            phynxInstance._sourceName = name
            self._sourceObjectList.append(phynxInstance)
        if FAMILY:
            pattern = get_family_pattern(self.__sourceNameList)
            if '%' in pattern:
                try:
                    phynxInstance = phynx.File(pattern, 'r',
                                            driver='family', 
                                            lock=None,
                                           sorted_with=h5py_sorting)
                except TypeError:
                    phynxInstance = phynx.File(pattern, 'r', 
                                            driver='family',lock=None)
            else:
                raise IOError, "Cannot set of HDF5 files"
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
            if DEBUG:
                print "Error key not in list "
            return {}
    
    def __getKeyInfo(self,key):
        try:
            index, entry = key.split(".")
            index = int(index)-1
            entry = int(entry)-1            
        except:
            #should we rise an error?
            if DEBUG:
                print "Error trying to interpret key =",key
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
        selection: a dictionnary generated via QNexusWidget
        """
        if selection is not None:
            if selection.has_key('sourcename'):
                filename  = selection['sourcename']
                entry     = selection['entry']
                fileIndex  = self.__sourceNameList.index(filename)
                phynxFile =  self._sourceObjectList[fileIndex]
                if entry == "/":
                    entryIndex = 0
                else:
                    entryIndex = phynxFile["/"].keys().index(entry[1:])
            else:
                key_split = key.split(".")
                fileIndex = int(key_split[0])-1
                phynxFile =  self._sourceObjectList[fileIndex]
                entryIndex = int(key_split[1])-1
                entry = phynxFile["/"].keys()[entryIndex] 
            actual_key = "%d.%d" % (fileIndex+1, entryIndex+1)
            if actual_key != key:
                if entry != "/":
                    print "Warning selection keys do not match"
        else:
            sourcekeys = self.getSourceInfo()['KeyList']
            #a key corresponds to an image        
            key_split= key.split(".")
            actual_key= "%d.%d" % (int(key_split[0]), int(key_split[1]))
            if actual_key not in sourcekeys:
                raise KeyError,"Key %s not in source keys" % actual_key
            raise NotImplemented, "Direct NXdata plot not implemented yet"        
        #create data object
        output = DataObject.DataObject()
        output.info = self.__getKeyInfo(actual_key)
        output.info['selection'] = selection
        if selection['selectiontype'].upper() in ["SCAN", "MCA"]:
            output.info['selectiontype'] = "1D"
        elif selection['selectiontype'] == "3D":
            output.info['selectiontype'] = "3D"
        elif selection['selectiontype'] == "2D":
            output.info['selectiontype'] = "2D"
            output.info['imageselection'] = True
        else:
            raise TypeError, "Unsupported selection type %s" % selection['selectiontype']
        if selection.has_key('LabelNames'):
            output.info['LabelNames'] = selection['LabelNames']
        elif selection.has_key('aliaslist'):
            output.info['LabelNames'] = selection['aliaslist']
        else:
            output.info['LabelNames'] = selection['cntlist']
        output.x = None
        output.y = None
        output.m = None
        output.data = None
        for cnt in ['y', 'x', 'm']:
            if not selection.has_key(cnt):
                continue
            if not len(selection[cnt]):
                continue
            path =  entry + selection['cntlist'][selection[cnt][0]]
            data = phynxFile[path]
            totalElements = 1
            for dim in data.shape:
                totalElements *= dim
            if totalElements < 1.0E8:
                try:
                    data = phynxFile[path].value
                except MemoryError:
                    pass
            if output.info['selectiontype'] == "1D":
                if len(data.shape) == 2:
                    if min(data.shape) == 1:
                        data = numpy.ravel(data)
                    else:
                        raise TypeError, "%s selection is not 1D" % cnt.upper()
                elif len(data.shape) > 2:
                    raise TypeError, "%s selection is not 1D" % cnt.upper()                
            if cnt == 'y':
                if output.info['selectiontype'] == "2D":
                    output.data = data
                else:
                    output.y = [data]
            elif cnt == 'x':
                #there can be more than one X except for 1D
                if output.info['selectiontype'] == "1D":
                    if len(selection[cnt]) > 1:
                        raise TypeError, "%s selection is not 1D" % cnt.upper()
                if output.x is None:
                    output.x = [data]
                if len(selection[cnt]) > 1:
                    for xidx in range(1, len(selection[cnt])):
                        path =  entry + selection['cntlist'][selection[cnt][xidx]]
                        data = phynxFile[path].value
                        output.x.append(data)
            elif cnt == 'm':
                #only one monitor
                output.m = [data]
        # MCA specific
        if selection['selectiontype'].upper() == "MCA":
            if not output.info.has_key('Channel0'):
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
                if not selection.has_key(key):
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
     raise TypeError,"Invalid Source Type, source type should be one of %s" % source_types.keys()
  
  return sourceClass(name)

        
if __name__ == "__main__":
    import time
    try:
        sourcename=sys.argv[1]
        key       =sys.argv[2]        
    except:
        print "Usage: EdfFileDataSource <file> <key>"
        sys.exit()
    #one can use this:
    obj = EdfFileDataSource(sourcename)
    #or this:
    obj = DataSource(sourcename)
    #data = obj.getData(key,selection={'pos':(10,10),'size':(40,40)})
    #data = obj.getDataObject(key,selection={'pos':None,'size':None})
    t0 = time.time()
    data = obj.getDataObject(key,selection=None)
    print "elapsed = ",time.time() - t0
    print "info = ",data.info
    if data.data is not None:
        print "data shape = ",data.data.shape
        print Numeric.ravel(data.data)[0:10]
    else:
        print data.y[0].shape
        print Numeric.ravel(data.y[0])[0:10]
    data = obj.getDataObject('1.1',selection=None)
    r = int(key.split('.')[-1])
    print " data[%d,0:10] = " % (r-1),data.data[r-1   ,0:10]
    print " data[0:10,%d] = " % (r-1),data.data[0:10, r-1]        

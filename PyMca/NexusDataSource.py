#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
import DataObject
import os
import types

try:
    import PyMca.phynx as phynx
except:
    #I should never reach here
    try:
        from xpaxs.io import phynx
    except ImportError:
        import phynx

SOURCE_TYPE = "HDF5"
DEBUG = 0

class NexusDataSource:
    def __init__(self,nameInput):
        if type(nameInput) == types.ListType:
            nameList = nameInput
        else:
            nameList = [nameInput]
        for name in nameList:
            if type(name) != types.StringType:
                raise TypeError,"Constructor needs string as first argument"            
        self.sourceName   = nameList
        self.sourceType = SOURCE_TYPE
        self.__sourceNameList = nameList
        self.refresh()

    def refresh(self):
        self._sourceObjectList=[]
        for name in self.__sourceNameList:
            self._sourceObjectList.append(phynx.File(name, 'r', lock=None))
        self.__lastKeyInfo = {}

    def getSourceInfo(self):
        """
        Returns information about the EdfFile object created by
        SetSource, to give application possibility to know about
        it before loading.
        Returns a dictionary with the key "KeyList" (list of all available keys
        in this source). Each element in "KeyList" has the form 'n1.n2' where
        n1 is the source number and n2 entry number in file starting at 1.
        """        
        return self.__getSourceInfo()
        
        
    def __getSourceInfo(self):
        SourceInfo={}
        SourceInfo["SourceType"]=SOURCE_TYPE
        SourceInfo["KeyList"]=[]
        i = 0
        for sourceObject in self._sourceObjectList:
            i+=1
            nEntries = len(sourceObject["/"].listnames())
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
            filename  = selection['sourcename']
            entry     = selection['entry']
            fileIndex  = self.__sourceNameList.index(filename)
            phynxFile =  self._sourceObjectList[fileIndex]
            entryIndex = phynxFile["/"].listnames().index(entry[1:])
            actual_key = "%d.%d" % (fileIndex+1, entryIndex+1)
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
            output.info['LabelNames'] = selection['cntlist']
            output.x = None
            output.y = None
            output.m = None
            output.data = None
            path =  entry + selection['cntlist'][selection['y'][0]]
            output.y = [phynxFile[path].value]
            if selection.has_key('x'):
                if len(selection['x']):
                    path = entry + selection['cntlist'][selection['x'][0]]
                    output.x = [phynxFile[path].value]
            if selection.has_key('m'):
                if len(selection['m']):
                    path = entry + selection['cntlist'][selection['m'][0]]
                    output.m = [phynxFile[path].value]
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
    import sys,time
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

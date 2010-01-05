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
import posixpath
import DataObject
try:
    from PyMca import NexusDataSource
except:
    import NexusDataSource
import posixpath
import numpy
DEBUG = 0    
SOURCE_TYPE = "HDF5Stack1D"

class HDF5Stack1D(DataObject.DataObject):
    def __init__(self, filelist, selection,
                       scanlist=None,
                       dtype=None):
        DataObject.DataObject.__init__(self)

        #the data type of the generated stack
        self.__dtype = dtype

        if filelist is not None:
            if selection is not None:
                self.loadFileList(filelist, selection, scanlist)

    def loadFileList(self, filelist, selection, scanlist=None):
        """
        loadFileList(self, filelist, y, scanlist=None, monitor=None, x=None)
        filelist is the list of file names belonging to the stack
        selection is a dictionary with the keys x, y, m.
        x        is the path to the x data (the channels) in the spectrum,
                 without the first level "directory". It is unused (for now).
        y        is the path to the 1D data (the counts) in the spectrum,
                 without the first level "directory"
        m        is the path to the normalizing data (I0 or whatever)
                 without the first level "directory".
        scanlist is the list of first level "directories" containing the 1D data
                 Example: The actual path has the form:
                 /whatever1/whatever2/counts
                 That means scanlist = ["/whatever"]
                 and               selection['y'] = "/whatever2/counts"
        """
        if type(filelist) == type(''):
            filelist = [filelist]

        # all the files in the same source
        hdfStack = NexusDataSource.NexusDataSource(filelist)

        #if there is more than one file, it is assumed all the files have
        #the same structure.
        tmpHdf = hdfStack._sourceObjectList[0]
        entryNames = tmpHdf["/"].keys()

        #built the selection in terms of HDF terms
        #for the time being, the x selection will be ignored but not the
        #monitor and only one y is taken
        ySelection = selection['y']
        if type(ySelection) == type([]):
            ySelection = ySelection[0]
        mSelection = selection['m']
        if type(mSelection) == type([]):
            if len(mSelection):
                mSelection = mSelection[0]
            else:
                mSelection = None
        else:
            mSelection = None
            
        if scanlist is None:
            #if the scanlist is None, it is assumed we are interested on all
            #the scans containing the selection, not that all the scans
            #contain the selection.
            scanlist = []
            if 0:
                JUST_KEYS = False
                #expect same entry names in the files
                for entry in entryNames:
                    path = "/"+entry + ySelection
                    dirname = posixpath.dirname(path)
                    base = posixpath.basename(path)
                    try:
                        if base in tmpHdf[dirname].keys():                        
                            scanlist.append(entry)
                    except:
                        pass
            else:
                JUST_KEYS = True
                #expect same structure in the files
                i = 0
                for entry in entryNames:
                    i += 1
                    path = "/"+entry + ySelection
                    dirname = posixpath.dirname(path)
                    base = posixpath.basename(path)
                    try:
                        if base in tmpHdf[dirname].keys():                        
                            scanlist.append("1.%d" % i)
                    except:
                        pass
        else:
            try:
                number, order = map(int, scanlist[0].split("."))
                JUST_KEYS = True
            except:
                JUST_KEYS = False
            if not JUST_KEYS:
                for scan in scanlist:
                    if scan not in entryNames:
                        raise ValueError, "Entry %s not in file"

        nFiles = len(filelist)
        nScans = len(scanlist)
        if not nScans:
            raise IOError, "No entry contains the required data"

        #Now is to decide the number of mca ...
        #I assume all the scans contain the same number of mca
        if JUST_KEYS:
            path = "/" + entryNames[int(scanlist[0].split(".")[-1])-1] + ySelection
            if mSelection is not None:
                mpath = "/" + entryNames[int(scanlist[0].split(".")[-1])-1] + mSelection
        else:
            path = "/" + scanlist[0] + ySelection
            if mSelection is not None:
                mpath = "/" + scanlist[0] + mSelection
        yDataset = tmpHdf[path]

        if self.__dtype is None:
            self.__dtype = yDataset.dtype

        #figure out the shape of the stack
        shape = yDataset.shape
        mcaIndex = selection.get('index', len(shape)-1)
        if mcaIndex == -1:
            mcaIndex = len(shape) - 1
        dim0, dim1, mcaDim = self.getDimensions(nFiles, nScans, shape,
                                                index=mcaIndex)
        try:
            self.data = numpy.zeros((dim0, dim1, mcaDim), self.__dtype)
            DONE = False
        except MemoryError:
            if (nFiles == 1) and (len(shape) == 3):
                print "Attempting dynamic loading"
                self.data = yDataset
                if mSelection is not None:
                    mDataset = tmpHdf[mpath]
                    self.monitor = mDataset
                DONE = True
            else:
                raise
        
        if not DONE:
            self.info["McaIndex"] = 2
            n = 0
            i_idx = dim0 * dim1

            if dim0 == 1:
                self.onBegin(dim1)
            else:
                self.onBegin(dim0)
            self.incrProgressBar=0
            for hdf in hdfStack._sourceObjectList:
                entryNames = hdf["/"].keys()
                for scan in scanlist:
                    if JUST_KEYS:
                        entryName = entryNames[int(scan.split(".")[-1])-1]
                        path = entryName + ySelection
                        if mSelection is not None:
                            mpath = entryName + mSelection
                            mDataset = hdf[mpath].value
                    else:
                        path = scan + ySelection
                        if mSelection is not None:
                            mpath = scan + mSelection
                            mDataset = hdf[mpath].value
                    yDataset = hdf[path].value
                    if mcaIndex != 0:
                        yDataset.shape = -1, mcaDim
                        if mSelection is not None:
                            case = -1
                            nMonitorData = 1
                            for  v in mDataset.shape:
                                nMonitorData *= v
                            if nMonitorData == yDataset.shape[0]:
                                mDataset.shape = yDataset.shape[0]
                                case = 0
                            elif nMonitorData == (yDataset.shape[0] * yDataset.shape[1]):
                                case = 1
                                mDataset.shape = yDataset.shape[0], yDataset.shape[1]
                            if case == -1:
                                raise ValueError, "I do not know how to handle this monitor data"
                        for mca in range(yDataset.shape[0]):
                            i = int(n/dim1)
                            j = n % dim1
                            if mSelection is not None:
                                if case == 0:
                                    self.data[i, j, :] = yDataset[mca,:]/mDataset[mca]
                                elif case == 1:
                                    self.data[i, j, :]  = yDataset[mca,:]/mDataset[mca, :]
                            else:
                                self.data[i, j, :] = yDataset[mca,:]
                            n += 1
                    else:
                        yDataset.shape = mcaDim, -1
                        if mSelection is not None:
                            case = -1
                            nMonitorData = 1
                            for  v in mDataset.shape:
                                nMonitorData *= v
                            if nMonitorData == yDataset.shape[1]:
                                mDataset.shape = yDataset.shape[1]
                                case = 0
                            #elif nMonitorData == (yDataset.shape[1] * yDataset.shape[2]):
                            #    case = 1
                            #    mDataset.shape = yDataset.shape[1], yDataset.shape[2]
                            if case == -1:
                                raise ValueError, "I do not know how to handle this monitor data"
                        for mca in range(yDataset.shape[1]):
                            i = int(n/dim1)
                            j = n % dim1
                            if mSelection is not None:
                                if case == 0:
                                    self.data[i, j, :] = yDataset[:,mca]/mDataset[mca]
                                elif case == 1:
                                    self.data[i, j, :]  = yDataset[:, mca]/mDataset[:, mca]
                            else:
                                self.data[i, j, :] = yDataset[:, mca]
                            n += 1
                    if dim0 == 1:
                        self.onProgress(j)
                if dim0 != 1:
                    self.onProgress(i)
            self.onEnd()
        else:
            self.info["McaIndex"] = mcaIndex


        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = filelist
        self.info["Size"]       = 1
        self.info["NumberOfFiles"] = 1
        if mcaIndex == 0:
            self.info["FileIndex"] = 1
        else:
            self.info["FileIndex"] = 0
        self.info['McaCalib'] = [ 0.0, 1.0, 0.0]
        self.info['Channel0'] = 0
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i+1,)
            self.info[key] = shape[i]


    def getDimensions(self, nFiles, nScans, shape, index=None):
        #some body may want to overwrite this
        """
        Returns the shape of the final stack as (Dim0, Dim1, Nchannels)
        """
        if index is None:
            index = -1
        if index == -1:
            index = len(shape) - 1
        if DEBUG:
            print "INDEX = ", index
        #figure out the shape of the stack
        if len(shape) == 0:
            #a scalar?
            raise ValueError, "Selection corresponds to a scalar"
        elif len(shape) == 1:
            #nchannels
            nMca = 1
        elif len(shape) == 2:
            if index == 0:
                #npoints x nchannels
                nMca = shape[1]
            else:
                #npoints x nchannels
                nMca = shape[0]
        elif len(shape) == 3:
            if index in [2, -1]:
                #dim1 x dim2 x nchannels
                nMca = shape[0] * shape[1]
            elif index == 0:
                nMca = shape[1] * shape[2]
            else:
                raise IndexError, "Only first and last dimensions handled"
        else:
            nMca = 1
            for i in range(len(shape)):
                if i == index:
                    continue
                nMca *= shape[i]
                
        mcaDim = shape[index]
        if DEBUG:
            print "nMca = ", nMca
            print "mcaDim = ", mcaDim

        # HDF allows to work directly from the files without loading
        # them into memory.
        if (nScans == 1) and (nFiles > 1):
            if nMca == 1:
                #specfile like case
                dim0 = nFiles
                dim1 = nMca * nScans # nScans is 1
            else:
                #ESRF EDF like case
                dim0 = nFiles
                dim1 = nMca * nScans # nScans is 1
        elif (nScans == 1) and (nFiles == 1):
            if nMca == 1:
                #specfile like single mca
                dim0 = nFiles # it is 1
                dim1 = nMca * nScans # nScans is 1
            elif len(shape) == 2:
                dim0 = nFiles # it is 1
                dim1 = nMca * nScans # nScans is 1
            elif len(shape) == 3:
                if index == 0:
                    dim0 = shape[1]
                    dim1 = shape[2]
                else:
                    dim0 = shape[0]
                    dim1 = shape[1]
            else:
                #specfile like multiple mca
                dim0 = nFiles # it is 1
                dim1 = nMca * nScans  # nScans is 1
        elif (nScans > 1)  and (nFiles == 1):
            if nMca == 1:
                #specfile like case
                dim0 = nFiles
                dim1 = nMca * nScans
            elif nMca > 1:
                if len(shape) == 1:
                    #specfile like case
                    dim0 = nFiles
                    dim1 = nMca * nScans
                elif len(shape) == 2:
                    dim0 = nScans
                    dim1 = nMca     #shape[0]
                elif len(shape) == 3:
                    if (shape[0] == 1) or (shape[1] == 1):
                        dim0 = nScans
                        dim1 = nMca
                    else:
                        #The user will have to decide the shape
                        dim0 = 1
                        dim1 = nScans * nMca
                else:
                    #The user will have to decide the shape
                    dim0 = 1
                    dim1 = nScans * nMca
        elif (nScans > 1) and (nFiles > 1):
            dim0 = nFiles
            dim1 = nMca * nScans
        else:
            #I should not reach this point
            raise ValueError, "Unhandled case"

        return dim0, dim1, shape[index]

    def onBegin(self, n):
        pass

    def onProgress(self, n):
        pass

    def onEnd(self):
        pass

if __name__ == "__main__":
    import sys

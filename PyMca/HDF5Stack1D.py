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
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import posixpath
import DataObject
try:
    from PyMca import NexusDataSource
except:
    import NexusDataSource
import posixpath
import numpy
    
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
        else:
            path = "/" + scanlist[0] + ySelection
        yDataset = tmpHdf[path] 
        if self.__dtype is None:
            self.__dtype = yDataset.dtype

        #figure out the shape of the stack
        shape = yDataset.shape

        dim0, dim1, mcaDim = self.getDimensions(nFiles, nScans, shape)
        try:
            self.data = numpy.zeros((dim0, dim1, mcaDim), self.__dtype)
            DONE = False
        except MemoryError:
            if (nFiles == 1) and (len(shape) == 3):
                print "Attempting dynamic loading"
                self.data = yDataset
                DONE = True
            else:
                raise
        
        if not DONE:
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
                    else:
                        path = scan + ySelection
                    yDataset = hdf[path].value
                    yDataset.shape = -1, mcaDim
                    for mca in range(yDataset.shape[0]):
                        i = int(n/dim1)
                        j = n % dim1
                        self.data[i, j, :] = yDataset[mca,:]
                        n += 1
                    if dim0 == 1:
                        self.onProgress(j)
                if dim0 != 1:
                    self.onProgress(i)
            self.onEnd()

        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = filelist[0]
        self.info["Size"]       = 1
        self.info["NumberOfFiles"] = 1
        self.info["FileIndex"] = 0
        self.info['McaCalib'] = [ 0.0, 1.0, 0.0]
        self.info['Channel0'] = 0
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i+1,)
            self.info[key] = shape[i]


    def getDimensions(self, nFiles, nScans, shape):
        #some body may want to overwrite this
        """
        Returns the shape of the final stack as (Dim0, Dim1, Nchannels)
        """
        #figure out the shape of the stack
        if len(shape) == 0:
            #a scalar?
            raise ValueError, "Selection corresponds to a scalar"
        elif len(shape) == 1:
            #nchannels
            nMca = 1
        elif len(shape) == 2:
            #npoints x nchannels
            nMca = shape[0]
        elif len(shape) == 3:
            #dim1 x dim2 x nchannels
            nMca = shape[0] * shape[1]
        else:
            #assume the last dimension is the mca
            nMca = 1
            for i in range(len(shape) - 1):
                nMca *= shape[i]

        mcaDim = shape[-1]

        # HDF allows to work directly from the files without loading
        # them into memory. I do not use that feature (yet)
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
                dim1 = shape[0] * nScans # nScans is 1
            elif len(shape) == 3:
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

        return dim0, dim1, shape[-1]

    def onBegin(self, n):
        pass

    def onProgress(self, n):
        pass

    def onEnd(self):
        pass

if __name__ == "__main__":
    import sys

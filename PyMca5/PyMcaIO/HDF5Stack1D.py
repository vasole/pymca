#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
import posixpath
import numpy
import h5py
try:
    from PyMca5.PyMcaCore import DataObject
    from PyMca5.PyMcaMisc import PhysicalMemory
except ImportError:
    print("HDF5Stack1D importing DataObject from local directory!")
    import DataObject
    import PhysicalMemory
try:
    from PyMca5.PyMcaCore import NexusDataSource
except ImportError:
    print("HDF5Stack1D importing NexusDataSource from local directory!")
    import NexusDataSource

DEBUG = 0
SOURCE_TYPE = "HDF5Stack1D"

class HDF5Stack1D(DataObject.DataObject):
    def __init__(self, filelist, selection,
                       scanlist=None,
                       dtype=None):
        DataObject.DataObject.__init__(self)

        #the data type of the generated stack
        self.__dtype0 = dtype
        self.__dtype  = dtype

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
                 That means scanlist = ["/whatever1"]
                 and               selection['y'] = "/whatever2/counts"
        """
        if DEBUG:
            print("filelist = ", filelist)
            print("selection = ", selection)
            print("scanlist = ", scanlist)
        # all the files in the same source
        hdfStack = NexusDataSource.NexusDataSource(filelist)

        #if there is more than one file, it is assumed all the files have
        #the same structure.
        tmpHdf = hdfStack._sourceObjectList[0]
        entryNames = []
        for key in tmpHdf["/"].keys():
            if isinstance(tmpHdf["/"+key], h5py.Group):
                entryNames.append(key)

        # built the selection in terms of HDF terms
        # for the time being, only the first item in x selection used

        xSelection = selection['x']
        if xSelection is not None:
            if type(xSelection) != type([]):
                xSelection = [xSelection]
        if type(xSelection) == type([]):
            if len(xSelection):
                xSelection = xSelection[0]
            else:
                xSelection = None
        else:
            xSelection = None
        # only one y is taken
        ySelection = selection['y']
        if type(ySelection) == type([]):
            ySelection = ySelection[0]

        # monitor selection
        mSelection = selection['m']
        if mSelection not in [None, []]:
            if type(mSelection) != type([]):
                mSelection = [mSelection]
        if type(mSelection) == type([]):
            if len(mSelection):
                mSelection = mSelection[0]
            else:
                mSelection = None
        else:
            mSelection = None

        USE_JUST_KEYS = False
        # deal with the pathological case where the scanlist corresponds
        # to a selected top level dataset
        if len(entryNames) == 0:
            if scanlist is not None:
                if (ySelection in scanlist) or \
                   (xSelection in scanlist) or \
                   (mSelection in scanlist):
                    scanlist = None
                    USE_JUST_KEYS = True
        elif len(entryNames) == 1:
            # deal with the SOLEIL case of one entry but with different name
            # in different files
            USE_JUST_KEYS = True
        elif scanlist in [None, []]:
            USE_JUST_KEYS = True
        if USE_JUST_KEYS:
            #if the scanlist is None, it is assumed we are interested on all
            #the scans containing the selection, not that all the scans
            #contain the selection.
            scanlist = []
            if 0:
                JUST_KEYS = False
                #expect same entry names in the files
                #Unfortunately this does not work for SOLEIL
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
                #expect same structure in the files even if the
                #names are different (SOLEIL ...)
                if len(entryNames):
                    i = 0
                    for entry in entryNames:
                        path = "/"+entry + ySelection
                        dirname = posixpath.dirname(path)
                        base = posixpath.basename(path)
                        if hasattr(tmpHdf[dirname], "keys"):
                            i += 1
                            if base in tmpHdf[dirname].keys():
                                scanlist.append("1.%d" % i)
                    if not len(scanlist):
                        path = "/" + ySelection
                        dirname = posixpath.dirname(path)
                        base = posixpath.basename(path)
                        try:
                            if base in tmpHdf[dirname].keys():
                                JUST_KEYS = False
                                scanlist.append("")
                        except:
                            #it will crash later on
                            pass
                else:
                    JUST_KEYS = False
                    scanlist.append("")
        else:
            try:
                number, order = [int(x) for x in scanlist[0].split(".")]
                JUST_KEYS = True
            except:
                JUST_KEYS = False
            if not JUST_KEYS:
                for scan in scanlist:
                    if scan.startswith("/"):
                        t = scan[1:]
                    else:
                        t = scan
                    if t not in entryNames:
                        raise ValueError("Entry %s not in file" % scan)

        nFiles = len(filelist)
        nScans = len(scanlist)
        if JUST_KEYS:
            if not nScans:
                raise IOError("No entry contains the required data")

        if DEBUG:
            print("Retained number of files = %d" % nFiles)
            print("Retained number of scans = %d" % nScans)

        #Now is to decide the number of mca ...
        #I assume all the scans contain the same number of mca
        if JUST_KEYS:
            path = "/" + entryNames[int(scanlist[0].split(".")[-1])-1] + ySelection
            if mSelection is not None:
                mpath = "/" + entryNames[int(scanlist[0].split(".")[-1])-1] + mSelection
            if xSelection is not None:
                xpath = "/" + entryNames[int(scanlist[0].split(".")[-1])-1] + xSelection
        else:
            path = scanlist[0] +  ySelection
            if mSelection is not None:
                mpath = scanlist[0] + mSelection
            if xSelection is not None:
                xpath = scanlist[0] + xSelection

        yDataset = tmpHdf[path]

        if self.__dtype is None:
            self.__dtype = yDataset.dtype
            if self.__dtype in [numpy.int16, numpy.uint16]:
                self.__dtype = numpy.float32
            elif self.__dtype in [numpy.int32, numpy.uint32]:
                self.__dtype = numpy.float64

        #figure out the shape of the stack
        shape = yDataset.shape
        mcaIndex = selection.get('index', len(shape)-1)
        if mcaIndex == -1:
            mcaIndex = len(shape) - 1
        if DEBUG:
            print("mcaIndex = %d" % mcaIndex)
        considerAsImages = False
        dim0, dim1, mcaDim = self.getDimensions(nFiles, nScans, shape,
                                                index=mcaIndex)
        try:
            if self.__dtype in [numpy.float32, numpy.int32]:
                bytefactor = 4
            elif self.__dtype in [numpy.int16, numpy.uint16]:
                bytefactor = 2
            elif self.__dtype in [numpy.int8, numpy.uint8]:
                bytefactor = 1
            else:
                bytefactor = 8

            neededMegaBytes = nFiles * dim0 * dim1 * (mcaDim * bytefactor/(1024*1024.))
            physicalMemory = PhysicalMemory.getPhysicalMemoryOrNone()
            if physicalMemory is None:
                # 5 Gigabytes should be a good compromise
                physicalMemory = 6000
            else:
                physicalMemory /= (1024*1024.)
            if (neededMegaBytes > (0.95*physicalMemory))\
               and (nFiles == 1) and (len(shape) == 3):
                if self.__dtype0 is None:
                    if (bytefactor == 8) and (neededMegaBytes < (2*physicalMemory)):
                        #try reading as float32
                        self.__dtype = numpy.float32
                    else:
                        raise MemoryError("Force dynamic loading")
                else:
                    raise MemoryError("Force dynamic loading")
            if (mcaIndex == 0) and ( nFiles == 1) and (nScans == 1):
                #keep the original arrangement but in memory
                self.data = numpy.zeros(yDataset.shape, self.__dtype)
                considerAsImages = True
            else:
                # force arrangement as spectra
                self.data = numpy.zeros((dim0, dim1, mcaDim), self.__dtype)
            DONE = False
        except (MemoryError, ValueError):
            #some versions report ValueError instead of MemoryError
            if (nFiles == 1) and (len(shape) == 3):
                print("Attempting dynamic loading")
                self.data = yDataset
                if mSelection is not None:
                    mDataset = tmpHdf[mpath].value
                    self.monitor = [mDataset]
                if xSelection is not None:
                    xDataset = tmpHdf[xpath].value
                    self.x = [xDataset]
                if h5py.version.version < '2.0':
                    #prevent automatic closing keeping a reference
                    #to the open file
                    self._fileReference = hdfStack
                DONE = True
            else:
                #what to do if the number of dimensions is only 2?
                raise

        if (not DONE) and (not considerAsImages):
            self.info["McaIndex"] = 2
            n = 0

            if dim0 == 1:
                self.onBegin(dim1)
            else:
                self.onBegin(dim0)
            self.incrProgressBar=0
            for hdf in hdfStack._sourceObjectList:
                entryNames = list(hdf["/"].keys())
                goodEntryNames = []
                for entry in entryNames:
                    tmpPath = "/" + entry
                    if hasattr(hdf[tmpPath], "keys"):
                        goodEntryNames.append(entry)
                for scan in scanlist:
                    if JUST_KEYS:
                        entryName = goodEntryNames[int(scan.split(".")[-1])-1]
                        path = entryName + ySelection
                        if mSelection is not None:
                            mpath = entryName + mSelection
                            mDataset = hdf[mpath].value
                        if xSelection is not None:
                            xpath = entryName + xSelection
                            xDataset = hdf[xpath].value
                    else:
                        path = scan + ySelection
                        if mSelection is not None:
                            mpath = scan + mSelection
                            mDataset = hdf[mpath].value
                        if xSelection is not None:
                            xpath = scan + xSelection
                            xDataset = hdf[xpath].value
                    try:
                        yDataset = hdf[path]
                        tmpShape = yDataset.shape
                        totalBytes = numpy.ones((1,), yDataset.dtype).itemsize
                        for nItems in tmpShape:
                            totalBytes *= nItems
                        if (totalBytes/(1024.*1024.)) > 500:
                            #read from disk
                            IN_MEMORY = False
                        else:
                            #read the data into memory
                            yDataset = hdf[path].value
                            IN_MEMORY = True
                    except (MemoryError, ValueError):
                        yDataset = hdf[path]
                        IN_MEMORY = False
                    nMcaInYDataset = 1
                    for dim in yDataset.shape:
                        nMcaInYDataset *= dim
                    nMcaInYDataset = int(nMcaInYDataset/mcaDim)
                    if mcaIndex != 0:
                        if IN_MEMORY:
                            yDataset.shape = -1, mcaDim
                        if mSelection is not None:
                            case = -1
                            nMonitorData = 1
                            for  v in mDataset.shape:
                                nMonitorData *= v
                            if nMonitorData == nMcaInYDataset:
                                mDataset.shape = nMcaInYDataset
                                case = 0
                            elif nMonitorData == (nMcaInYDataset * mcaDim):
                                case = 1
                                mDataset.shape = nMcaInYDataset, mcaDim
                            if case == -1:
                                raise ValueError(\
                                    "I do not know how to handle this monitor data")
                        if (len(yDataset.shape) == 3) and\
                           (dim1 == yDataset.shape[1]):
                            mca = 0
                            deltaI = int(yDataset.shape[1]/dim1)
                            for ii in range(yDataset.shape[0]):
                                i = int(n/dim1)
                                yData = yDataset[ii:(ii+1)]
                                yData.shape = -1, mcaDim
                                if mSelection is not None:
                                    if case == 0:
                                        mData = numpy.outer(mDataset[mca:(mca+dim1)],
                                                            numpy.ones((mcaDim)))
                                        self.data[i, :, :] = yData/mData
                                    elif case == 1:
                                        mData = mDataset[mca:(mca+dim1), :]
                                        mData.shape = -1, mcaDim
                                        self.data[i, :, :]  = yData/mData
                                else:
                                    self.data[i:(i+deltaI), :] = yData
                                n += yDataset.shape[1]
                                mca += dim1
                        else:
                            for mca in range(nMcaInYDataset):
                                i = int(n/dim1)
                                j = n % dim1
                                if len(yDataset.shape) == 3:
                                    ii = int(mca/yDataset.shape[1])
                                    jj = mca % yDataset.shape[1]
                                    yData = yDataset[ii, jj]
                                elif len(yDataset.shape) == 2:
                                    yData = yDataset[mca,:]
                                elif len(yDataset.shape) == 1:
                                    yData = yDataset
                                if mSelection is not None:
                                    if case == 0:
                                        self.data[i, j, :] = yData/mDataset[mca]
                                    elif case == 1:
                                        self.data[i, j, :]  = yData/mDataset[mca, :]
                                else:
                                    self.data[i, j, :] = yData
                                n += 1
                    else:
                        if mSelection is not None:
                            case = -1
                            nMonitorData = 1
                            for  v in mDataset.shape:
                                nMonitorData *= v
                            if nMonitorData == yDataset.shape[0]:
                                case = 3
                                mDataset.shape = yDataset.shape[0]
                            elif nMonitorData == nMcaInYDataset:
                                mDataset.shape = nMcaInYDataset
                                case = 0
                            #elif nMonitorData == (yDataset.shape[1] * yDataset.shape[2]):
                            #    case = 1
                            #    mDataset.shape = yDataset.shape[1], yDataset.shape[2]
                            if case == -1:
                                raise ValueError(\
                                    "I do not know how to handle this monitor data")
                        if IN_MEMORY:
                            yDataset.shape = mcaDim, -1
                        if len(yDataset.shape) != 3:
                            for mca in range(nMcaInYDataset):
                                i = int(n/dim1)
                                j = n % dim1
                                if len(yDataset.shape) == 3:
                                    ii = int(mca/yDataset.shape[2])
                                    jj = mca % yDataset.shape[2]
                                    yData = yDataset[:, ii, jj]
                                elif len(yDataset.shape) == 2:
                                    yData = yDataset[:, mca]
                                elif len(yDataset.shape) == 1:
                                    yData = yDataset[:]
                                if mSelection is not None:
                                    if case == 0:
                                        self.data[i, j, :] = yData/mDataset[mca]
                                    elif case == 1:
                                        self.data[i, j, :]  = yData/mDataset[:, mca]
                                    elif case == 3:
                                        self.data[i, j, :]  = yData/mDataset
                                else:
                                    self.data[i, j, :] = yData
                                n += 1
                        else:
                            #stack of images to be read as MCA
                            for nImage in range(yDataset.shape[0]):
                                tmp = yDataset[nImage:(nImage+1)]
                                if len(tmp.shape) == 3:
                                    i = int(n/dim1)
                                    j = n % dim1
                                    if 0:
                                        #this loop is extremely SLOW!!!(and useless)
                                        for ii in range(tmp.shape[1]):
                                            for jj in range(tmp.shape[2]):
                                                self.data[i+ii, j+jj, nImage] = tmp[0, ii, jj]
                                    else:
                                        self.data[i:i+tmp.shape[1],
                                                  j:j+tmp.shape[2], nImage] = tmp[0]
                            if mSelection is not None:
                                for mca in range(yDataset.shape[0]):
                                    i = int(n/dim1)
                                    j = n % dim1
                                    yData = self.data[i, j, :]
                                    if case == 0:
                                        self.data[i, j, :] = yData/mDataset[mca]
                                    elif case == 1:
                                        self.data[i, j, :]  = yData/mDataset[:, mca]
                                    n += 1
                            else:
                                n += tmp.shape[1] * tmp.shape[2]
                    if dim0 == 1:
                        self.onProgress(j)
                if dim0 != 1:
                    self.onProgress(i)
            self.onEnd()
        elif not DONE:
            # data into memory but as images
            self.info["McaIndex"] = mcaIndex
            for hdf in hdfStack._sourceObjectList:
                entryNames = list(hdf["/"].keys())
                for scan in scanlist:
                    if JUST_KEYS:
                        entryName = entryNames[int(scan.split(".")[-1])-1]
                        path = entryName + ySelection
                        if mSelection is not None:
                            mpath = entryName + mSelection
                            mDataset.shape
                        if xSelection is not None:
                            xpath = entryName + xSelection
                            xDataset = hdf[xpath].value
                    else:
                        path = scan + ySelection
                        if mSelection is not None:
                            mpath = scan + mSelection
                            mDataset = hdf[mpath].value
                        if xSelection is not None:
                            xpath = scan + xSelection
                            xDataset = hdf[xpath].value
                    if mSelection is not None:
                        nMonitorData = mDataset.size
                        case = -1
                        yDatasetShape = yDataset.shape
                        if nMonitorData == yDatasetShape[0]:
                            #as many monitor data as images
                            mDataset.shape = yDatasetShape[0]
                            case = 0
                        elif nMonitorData == (yDatasetShape[1] * yDatasetShape[2]):
                            #as many monitorData as pixels
                            case = 1
                            mDataset.shape = yDatasetShape[1], yDatasetShape[2]
                        if case == -1:
                            raise ValueError(\
                                "I do not know how to handle this monitor data")
                        if case == 0:
                            for i in range(yDatasetShape[0]):
                                self.data[i] = yDataset[i].value / mDataset[i]
                        elif case == 1:
                            for i in range(yDataset.shape[0]):
                                self.data[i] = yDataset[i] / mDataset
                    else:
                        for i in range(yDataset.shape[0]):
                            self.data[i:i+1] = yDataset[i:i+1]
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
        if xSelection is not None:
            if xDataset.size == shape[self.info['McaIndex']]:
                self.x = [xDataset.reshape(-1)]
            else:
                print("Ignoring xSelection")


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
            print("INDEX = %d" % index)
        #figure out the shape of the stack
        if len(shape) == 0:
            #a scalar?
            raise ValueError("Selection corresponds to a scalar")
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
                raise IndexError("Only first and last dimensions handled")
        else:
            nMca = 1
            for i in range(len(shape)):
                if i == index:
                    continue
                nMca *= shape[i]

        mcaDim = shape[index]
        if DEBUG:
            print("nMca = %d" % nMca)
            print("mcaDim = ", mcaDim)

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
            raise ValueError("Unhandled case")

        return dim0, dim1, shape[index]

    def onBegin(self, n):
        pass

    def onProgress(self, n):
        pass

    def onEnd(self):
        pass

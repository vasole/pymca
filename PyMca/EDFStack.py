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
import DataObject
import EdfFile
import EdfFileDataSource
import numpy.oldnumeric as Numeric
import numpy
import sys
import os
HDF5 = False
try:
    import h5py
    import PyMca.phynx as phynx
    HDF5 = True
except ImportError:
    try:
        import phynx
        HDF5 = True
    except ImportError:
        try:
            from xpaxs.io import phynx
            HDF5 = True
        except ImportError:
            pass
        

SOURCE_TYPE = "EdfFileStack"
DEBUG = 0

X_AXIS=0
Y_AXIS=1
Z_AXIS=2

class EDFStack(DataObject.DataObject):
    def __init__(self, filelist = None, imagestack=None, dtype=None):
        DataObject.DataObject.__init__(self)
        self.incrProgressBar=0
        self.__keyList = []
        if imagestack is None:
            self.__imageStack = False
        else:
            self.__imageStack = imagestack
        self.__dtype = dtype
        if filelist is not None:
            if type(filelist) != type([]):
                filelist = [filelist]
            if len(filelist) == 1:
                self.loadIndexedStack(filelist)
            else:
                self.loadFileList(filelist)

    def loadFileList(self, filelist, fileindex=0):
        if type(filelist) == type(''):filelist = [filelist]
        self.__keyList = []
        self.sourceName = filelist
        self.__indexedStack = True
        self.sourceType = SOURCE_TYPE
        self.info = {}
        self.nbFiles=len(filelist)

        #read first edf file
        #get information
        tempEdf=EdfFileDataSource.EdfFileDataSource(filelist[0])
        keylist = tempEdf.getSourceInfo()['KeyList']
        nImages = len(keylist)
        dataObject = tempEdf.getDataObject(keylist[0])
        self.info.update(dataObject.info)
        if len(dataObject.data.shape) == 3:
            #this is already a stack
            self.data = dataObject.data
            self.__nFiles         = 1
            self.__nImagesPerFile = nImages
            shape = self.data.shape
            for i in range(len(shape)):
                key = 'Dim_%d' % (i+1,)
                self.info[key] = shape[i]
            self.info["SourceType"] = SOURCE_TYPE
            self.info["SourceName"] = filelist[0]
            self.info["Size"]       = 1
            self.info["NumberOfFiles"] = 1
            self.info["FileIndex"] = fileindex
            return
        arrRet = dataObject.data
        if self.__dtype is None:
            self.__dtype = arrRet.dtype

        self.onBegin(self.nbFiles)
        singleImageShape = arrRet.shape
        if (fileindex == 2) or (self.__imageStack):
            self.__imageStack = True
            if len(singleImageShape) == 1:
                #single line
                #be ready for specfile stack?
                raise IOError, "Not implemented yet"
                self.data = numpy.zeros((arrRet.shape[0],
                                           nImages,
                                           self.nbFiles),
                                           self.__dtype)
                self.incrProgressBar=0
                for tempEdfFileName in filelist:
                    tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                    for i in range(nImages):
                        pieceOfStack=tempEdf.GetData(i)
                        self.data[:,i, self.incrProgressBar] = pieceOfStack[:]
                    self.incrProgressBar += 1
                    self.onProgress(self.incrProgressBar)
                self.onEnd()
            else:
                if nImages > 1:
                    #this is not the common case
                    #should I try to convert it to a standard one
                    #using a 3D matrix or keep as 4D matrix?
                    if self.nbFiles > 1:
                        raise IOError, "Multiple files with multiple images implemented yet"
                    self.data = numpy.zeros((arrRet.shape[0],
                                               arrRet.shape[1],
                                               nImages * self.nbFiles),
                                               self.__dtype)
                    self.incrProgressBar=0
                    for tempEdfFileName in filelist:
                        tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                        for i in range(nImages):
                            pieceOfStack=tempEdf.GetData(i)
                            self.data[:,:,
                                      nImages*self.incrProgressBar+i] = \
                                                      pieceOfStack[:,:]
                        self.incrProgressBar += 1
                else:
                    #this is the common case
                    try:
                        self.data = numpy.zeros((arrRet.shape[0],
                                                 arrRet.shape[1],
                                                 self.nbFiles),
                                                 self.__dtype)
                        self.incrProgressBar=0
                        for tempEdfFileName in filelist:
                            tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                            pieceOfStack=tempEdf.GetData(0)    
                            self.data[:,:, self.incrProgressBar] = pieceOfStack
                            self.incrProgressBar += 1
                            self.onProgress(self.incrProgressBar)
                    except (MemoryError, ValueError):
                        hdf5done = False
                        if HDF5 and ('PyMcaQt' in sys.modules):
                            import PyMcaQt as qt
                            import ArraySave
                            msg=qt.QMessageBox.information( None,
                              "Memory error\n",
                              "Do you want to convert your data to HDF5?\n",
                              qt.QMessageBox.Yes,qt.QMessageBox.No)
                            if msg != qt.QMessageBox.No:
                                hdf5file = qt.QFileDialog.getSaveFileName(None,
                                            "Please select output file name",
                                            os.path.dirname(filelist[0]),
                                            "HDF5 files *.h5")
                                if not len(hdf5file):
                                    raise IOError, "Invalid output file"
                                hdf5file = str(hdf5file)
                                if not hdf5file.endswith(".h5"):
                                    hdf5file += ".h5"
                                hdf, self.data =  ArraySave.getHDF5FileInstanceAndBuffer(hdf5file,
                                              (self.nbFiles,
                                               arrRet.shape[0],
                                               arrRet.shape[1]))
                                self.incrProgressBar=0
                                for tempEdfFileName in filelist[0:10]:
                                    tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                                    pieceOfStack=tempEdf.GetData(0)
                                    self.data[self.incrProgressBar,:,:] = pieceOfStack[:,:]
                                    hdf.flush()
                                    self.incrProgressBar += 1
                                    self.onProgress(self.incrProgressBar)
                                hdf5done = True
                        if not hdf5done:
                            for i in range(3):
                                print "\7"
                            samplingStep = None
                            i = 2
                            while samplingStep is None:
                                print "**************************************************"
                                print " Memory error!, attempting %dx%d sampling reduction " % (i,i)
                                print "**************************************************"
                                s1, s2 = arrRet[::i, ::i].shape
                                try:
                                    self.data = numpy.zeros((s1, s2,
                                                         self.nbFiles),
                                                         self.__dtype)
                                    samplingStep = i
                                except:
                                    i += 1
                            self.incrProgressBar=0
                            for tempEdfFileName in filelist:
                                tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                                pieceOfStack=tempEdf.GetData(0)
                                self.data[:,:, self.incrProgressBar] = pieceOfStack[
                                                            ::samplingStep,::samplingStep]
                                self.incrProgressBar += 1
                                self.onProgress(self.incrProgressBar)
                    self.onEnd()
        else:
            self.__imageStack = False
            if len(singleImageShape) == 1:
                #single line
                #be ready for specfile stack?
                raise IOError, "Not implemented yet"
                self.data = numpy.zeros((self.nbFiles,
                                           arrRet.shape[0],
                                           nImages),
                                           self.__dtype)
                self.incrProgressBar=0
                for tempEdfFileName in filelist:
                    tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                    for i in range(nImages):
                        pieceOfStack=tempEdf.GetData(i)
                        self.data[self.incrProgressBar, :,i] = pieceOfStack[:]
                    self.incrProgressBar += 1
                    self.onProgress(self.incrProgressBar)
                self.onEnd()
            else:
                if nImages > 1:
                    #this is not the common case
                    #should I try to convert it to a standard one
                    #using a 3D matrix or kepp as 4D matrix?
                    if self.nbFiles > 1:
                        if (arrRet.shape[0] > 1) and\
                           (arrRet.shape[1] > 1):
                                raise IOError, "Multiple files with multiple images not implemented yet"
                        elif arrRet.shape[0] == 1:
                            self.data = numpy.zeros((self.nbFiles,
                                               arrRet.shape[0] * nImages,
                                               arrRet.shape[1]),
                                               self.__dtype)
                            self.incrProgressBar=0
                            for tempEdfFileName in filelist:
                                tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                                for i in range(nImages):
                                    pieceOfStack=tempEdf.GetData(i)
                                    self.data[self.incrProgressBar,
                                              i,:] = \
                                                              pieceOfStack[:,:]
                                self.incrProgressBar += 1
                                self.onProgress(self.incrProgressBar)
                        elif arrRet.shape[1] == 1:
                            self.data = numpy.zeros((self.nbFiles,
                                               arrRet.shape[1] * nImages,
                                               arrRet.shape[0]),
                                               self.__dtype)
                            self.incrProgressBar=0
                            for tempEdfFileName in filelist:
                                tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                                for i in range(nImages):
                                    pieceOfStack=tempEdf.GetData(i)
                                    self.data[self.incrProgressBar,
                                              i,:] = \
                                                              pieceOfStack[:,:]
                                self.incrProgressBar += 1
                                self.onProgress(self.incrProgressBar)
                    else:
                        self.data = numpy.zeros((nImages * self.nbFiles,
                                               arrRet.shape[0],
                                               arrRet.shape[1]),
                                               self.__dtype)
                        self.incrProgressBar=0
                        for tempEdfFileName in filelist:
                            tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                            for i in range(nImages):
                                pieceOfStack=tempEdf.GetData(i)
                                self.data[nImages*self.incrProgressBar+i,
                                          :,:] = \
                                                          pieceOfStack[:,:]
                            self.incrProgressBar += 1
                            self.onProgress(self.incrProgressBar)
                    self.onEnd()
                else:
                    #this is the common case
                    if self.__dtype == numpy.float:
                        bytefactor = 8
                    else:
                        bytefactor = 4
                        
                    needed_ = self.nbFiles * \
                                   arrRet.shape[0] *\
                                   arrRet.shape[1] * bytefactor/(1024*1024)
                    if (needed_ > 2000) and (sys.platform == 'linux2'):
                        swapfile = '/tmp/pymcaroitool.dat'
                        swapdir = '/buffer/%s1' %  os.getenv('HOSTNAME')
                        if os.path.isdir(swapdir):
                            swapfile = '/buffer/%s1/pymcaroitool.dat' %\
                                            os.getenv('HOSTNAME')
                        print "needed megabytes= ", needed_
                        print "using a buffer: %s" % swapfile
                        if os.path.exists(swapfile):
                            os.remove(swapfile)

                        self.data = numpy.memmap(swapfile,
                                    shape= (self.nbFiles, arrRet.shape[0],
                                            arrRet.shape[1]),
                                    dtype= self.__dtype,
                                    mode='w+')
                        os.system(('chmod 777 %s' % swapfile))
                    else:
                        if fileindex == 1:
                            try:
                                self.data = numpy.zeros((arrRet.shape[0],
                                                        self.nbFiles,
                                                       arrRet.shape[1]),
                                                       self.__dtype)
                            except:
                                try:
                                    self.data = numpy.zeros((arrRet.shape[0],
                                                        self.nbFiles,
                                                       arrRet.shape[1]),
                                                       numpy.float32)
                                except:
                                    self.data = numpy.zeros((arrRet.shape[0],
                                                        self.nbFiles,
                                                       arrRet.shape[1]),
                                                       numpy.int16)
                        else:
                            try:
                                self.data = numpy.zeros((self.nbFiles,
                                                       arrRet.shape[0],
                                                       arrRet.shape[1]),
                                                       self.__dtype)
                            except:
                                try:
                                    self.data = numpy.zeros((self.nbFiles,
                                                       arrRet.shape[0],
                                                       arrRet.shape[1]),
                                                       numpy.float32)
                                except (MemoryError, ValueError):
                                    text = "Memory Error: Attempt subsampling or convert to HDF5"
                                    if HDF5 and ('PyMcaQt' in sys.modules):
                                        import PyMcaQt as qt
                                        import ArraySave
                                        msg=qt.QMessageBox.information( None,
                                          "Memory error\n",
                                          "Do you want to convert your data to HDF5?\n",
                                          qt.QMessageBox.Yes,qt.QMessageBox.No)
                                        if msg == qt.QMessageBox.No:
                                            raise MemoryError, text
                                        hdf5file = qt.QFileDialog.getSaveFileName(None,
                                                    "Please select output file name",
                                                    os.path.dirname(filelist[0]),
                                                    "HDF5 files *.h5")
                                        if not len(hdf5file):
                                            raise IOError, "Invalid output file"
                                        hdf5file = str(hdf5file)
                                        if not hdf5file.endswith(".h5"):
                                            hdf5file += ".h5"
                                        hdf, self.data =  ArraySave.getHDF5FileInstanceAndBuffer(hdf5file,
                                                      (self.nbFiles,
                                                       arrRet.shape[0],
                                                       arrRet.shape[1]))               
                                    else:    
                                        raise MemoryError, "Memory Error"
                    self.incrProgressBar=0
                    if fileindex == 1:
                        for tempEdfFileName in filelist:
                            tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                            pieceOfStack=tempEdf.GetData(0)    
                            self.data[:,self.incrProgressBar,:] = pieceOfStack[:,:]
                            self.incrProgressBar += 1
                            self.onProgress(self.incrProgressBar)
                    else:
                        for tempEdfFileName in filelist:
                            tempEdf=EdfFile.EdfFile(tempEdfFileName, 'rb')
                            pieceOfStack=tempEdf.GetData(0)
                            self.data[self.incrProgressBar, :,:] = pieceOfStack[:,:]
                            self.incrProgressBar += 1
                            self.onProgress(self.incrProgressBar)
                    self.onEnd()
        self.__nFiles         = self.incrProgressBar
        self.__nImagesPerFile = nImages
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i+1,)
            self.info[key] = shape[i]
        if not isinstance(self.data, numpy.ndarray):
            hdf.flush()
            self.info["SourceType"] = "HDF5Stack1D"
            if self.__imageStack:
                self.info["McaIndex"] = 0
                self.info["FileIndex"] = 1
            else:
                self.info["McaIndex"] = 2
                self.info["FileIndex"] = 0
            self.info["SourceName"] = [hdf5file]
            self.info["NumberOfFiles"] = 1
            self.info["Size"]       = 1
        else:
            self.info["SourceType"] = SOURCE_TYPE
            self.info["FileIndex"] = fileindex
            self.info["SourceName"] = self.sourceName
            self.info["NumberOfFiles"] = self.__nFiles * 1
            self.info["Size"] = self.__nFiles * self.__nImagesPerFile
            

    def onBegin(self, n):
        pass

    def onProgress(self, n):
        pass

    def onEnd(self):
        pass

    def loadIndexedStack(self,filename,begin=None,end=None, skip = None, fileindex=0):
        #if begin is None: begin = 0
        if type(filename) == type([]):
            filename = filename[0]
        if not os.path.exists(filename):
            raise IOError,"File %s does not exists" % filename
        name = os.path.basename(filename)
        n = len(name)
        i = 1
        numbers = ['0', '1', '2', '3', '4', '5',
                   '6', '7', '8','9']
        while (i <= n):
            c = name[n-i:n-i+1]
            if c in ['0', '1', '2',
                                '3', '4', '5',
                                '6', '7', '8',
                                '9']:
                break
            i += 1
        suffix = name[n-i+1:]
        if len(name) == len(suffix):
            #just one file, one should use standard widget
            #and not this one.
            self.loadFileList(filename, fileindex=fileindex)
        else:
            nchain = []
            while (i<=n):
                c = name[n-i:n-i+1]
                if c not in ['0', '1', '2',
                                    '3', '4', '5',
                                    '6', '7', '8',
                                    '9']:
                    break
                else:
                    nchain.append(c)
                i += 1
            number = ""
            nchain.reverse()
            for c in nchain:
                number += c
            format = "%" + "0%dd" % len(number)
            if (len(number) + len(suffix)) == len(name):
                prefix = ""
            else:
                prefix = name[0:n-i+1]
            prefix = os.path.join(os.path.dirname(filename),prefix)
            if not os.path.exists(prefix + number + suffix):
                print "Internal error in EDFStack"
                print "file should exist:",prefix + number + suffix
                return
            i = 0
            if begin is None:
                begin = 0
                testname = prefix+format % begin+suffix
                while not os.path.exists(prefix+format % begin+suffix):
                    begin += 1
                    testname = prefix+format % begin+suffix
                    if len(testname) > len(filename):break
                i = begin
            else:
                i = begin
            if not os.path.exists(prefix+format % i+suffix):
                raise ValueError,"Invalid start index file = %s" % \
                      prefix+format % i+suffix
            f = prefix+format % i+suffix
            filelist = []
            while os.path.exists(f):
                filelist.append(f)
                i += 1
                if end is not None:
                    if i > end:
                        break
                f = prefix+format % i+suffix
            self.loadFileList(filelist, fileindex=fileindex)

    def getSourceInfo(self):
        sourceInfo = {}
        sourceInfo["SourceType"]=SOURCE_TYPE
        if self.__keyList == []:
            for i in range(1, self.__nFiles + 1):
                for j in range(1, self.__nImages + 1):
                    self.__keyList.append("%d.%d" % (i,j))
        sourceInfo["KeyList"]= self.__keyList

    def getKeyInfo(self, key):
        print "Not implemented"
        return {}

    def isIndexedStack(self):
        return self.__indexedStack
    
    def getZSelectionArray(self,z=0):
        return (self.data[:,:,z]).astype(numpy.float)
        
    def getXYSelectionArray(self,coord=(0,0)):
        x,y=coord    
        return (self.data[y,x,:]).astype(numpy.float)

if __name__ == "__main__":
    import time
    t0= time.time()
    stack = EDFStack()
    #stack.loadIndexedStack("Z:\COTTE\ch09\ch09__mca_0005_0000_0070.edf")
    stack.loadIndexedStack(".\COTTE\ch09\ch09__mca_0005_0000_0070.edf")
    shape = stack.data.shape
    print "elapsed = ", time.time() - t0
    #guess the MCA
    imax = 0
    for i in range(len(shape)):
        if shape[i] > shape[imax]:
            imax = i

    print "selections ",
    print "getZSelectionArray  shape = ", stack.getZSelectionArray().shape
    print "getXYSelectionArray shape = ", stack.getXYSelectionArray().shape

    import PyMcaQt as qt
    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    if 1:
        import RGBCorrelatorGraph
        w = RGBCorrelatorGraph.RGBCorrelatorGraph()
        graph = w.graph
    else:
        import QtBlissGraph
        w = QtBlissGraph.QtBlissGraph()
        graph = w
    print "shape sum 0 = ",Numeric.sum(stack.data, 0).shape
    print "shape sum 1 = ",Numeric.sum(stack.data, 1).shape
    print "shape sum 2 = ",Numeric.sum(stack.data, 2).shape
    a = Numeric.sum(stack.data, imax)
    print a.shape
    graph.setX1AxisLimits(0, a.shape[0])
    if 0:
        w.setY1AxisLimits(0, a.shape[1])
        w.setY1AxisInverted(True)
    else:
        graph.setY1AxisInverted(True)
        graph.setY1AxisLimits(0, a.shape[1])
    graph.imagePlot(a, ymirror=0)
    if imax == 0:
        graph.x1Label('Column Number')
    else:
        graph.x1Label('Row Number')
    graph.ylabel('File Number')
    w.show()

    if imax == 0:
        mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),1)
    else:
        mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),0)

    import McaWindow
    mca = McaWindow.McaWidget()
    sel = {}
    sel['SourceName'] = "EDF Stack"
    sel['Key']        = "SUM"
    sel['legend']     = "EDF Stack SUM"
    mcaData = DataObject.DataObject()
    mcaData.info = {'McaCalib': [0 , 2.0 ,0],
                    "selectiontype":"1D",
                    "SourceName":"EDF Stack",
                    "Key":"SUM"}
    mcaData.x = [numpy.arange(len(mcaData0)).astype(numpy.float)]
    mcaData.y = [mcaData0]
    sel['dataobject'] = mcaData
    mca.show()
    mca._addSelection([sel])
    graph.enableSelection(True)
    def graphSlot(ddict):
        if ddict['event'] == "MouseSelection":
            ix1 = int(ddict['xmin'])
            ix2 = int(ddict['xmax'])+1
            iy1 = int(ddict['xmin'])
            iy2 = int(ddict['xmax'])+1
            if imax == 0:
                selectedData = Numeric.sum(Numeric.sum(stack.data[:,ix1:ix2, iy1:iy2], 2),1)
            else:
                selectedData = Numeric.sum(Numeric.sum(stack.data[ix1:ix2,:, iy1:iy2], 2),0)
            sel = {}
            sel['SourceName'] = "EDF Stack"
            sel['Key'] = "Selection"
            sel["selectiontype"] = "1D"
            sel['legend'] = "EDF Stack Selection"
            selDataObject = DataObject.DataObject()
            selDataObject.info={'McaCalib': [100 , 2.0 ,0],
                                "selectiontype":"1D",
                                "SourceName":"EDF Stack Selection",
                                "Key":"Selection"}
            selDataObject.x = [numpy.arange(len(mcaData0)).astype(numpy.float)]
            selDataObject.y = [selectedData]
            sel['dataobject'] = selDataObject
            mca._addSelection([sel])
    qt.QObject.connect(graph, qt.SIGNAL('QtBlissGraphSignal'),
                       graphSlot)
    #w.replot()
    if qt.qVersion() < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()

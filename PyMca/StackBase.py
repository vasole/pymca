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
"""
Base class to handle stacks.

"""
from PyMca import DataObject
import numpy
import time
import os
import sys
import glob
DEBUG = 0
PLUGINS_DIR = None
try:
    if os.path.exists(os.path.join(os.path.dirname(__file__),"PyMcaPlugins")):
        import PyMcaPlugins
        PLUGINS_DIR = os.path.dirname(PyMcaPlugins.__file__)
    else:
        directory = os.path.dirname(__file__)
        while True:
            if os.path.exists(os.path.join(directory, "PyMcaPlugins")):
                PLUGINS_DIR = os.path.join(directory, "PyMcaPlugins")
                break
            directory = os.path.dirname(directory)
            if len(directory) < 5:
                break
except:
    pass


class StackBase(object):
    def __init__(self):
        self._stack = DataObject.DataObject()
        self._stackImageData = None
        self._selectionMask  = None
        self._ROIDict = {'name': "ICR",
                         'type': "CHANNEL",
                         'calibration':[0, 1.0, 0.0],
                         'from': 0,
                         'to': -1}

        self._ROIImageDict = {'ROI': None,
                              'Maximum': None,
                              'Minimum': None,
                              'Left': None,
                              'Middle': None,
                              'Right': None,
                              'Background':None}

        self._ROIImageList = []
        self._ROIImageNames = []

        self.pluginList = []
        self.pluginInstanceDict = {}
        self.getPlugins()

    def getPlugins(self):
        """
        Import or reloads all the available plugins.
        It returns the number of plugins loaded.
        """
        if PLUGINS_DIR is None:
            return 0
        directory = PLUGINS_DIR
        if not os.path.exists(directory):
            raise IOError, "Directory:\n%s\ndoes not exist." % directory

        self.pluginList = []
        fileList = glob.glob(os.path.join(directory, "*.py"))
        targetMethod = 'getStackPluginInstance'
        for module in fileList:
            try:
                pluginName = os.path.basename(module)[:-3]
                plugin = "PyMcaPlugins." + pluginName
                if pluginName in self.pluginList:
                    idx = self.pluginList.index(pluginName)
                    del self.pluginList[idx]
                if plugin in self.pluginInstanceDict.keys():
                    del self.pluginInstanceDict[plugin]
                if plugin in sys.modules:
                    if hasattr(sys.modules[plugin], targetMethod):
                        reload(sys.modules[plugin])
                else:
                    __import__(plugin)
                if hasattr(sys.modules[plugin], targetMethod):
                    self.pluginInstanceDict[plugin] = \
                            sys.modules[plugin].getStackPluginInstance(self)
                    self.pluginList.append(plugin)
            except:
                if DEBUG:
                    print "Problem importing module %s" % plugin
        return len(self.pluginList)

    def setStack(self, stack, mcaindex=2, fileindex=None):
        if isinstance(stack, DataObject.DataObject):
            self._stack = stack
            self._stack.info['SourceName'] = stack.info.get('SourceName',
                                                            "Data of unknown origin")
        else:
            self._stack.data = stack
            self._stack.info['SourceName'] = "Data of unknown origin"

        info = self._stack.info
        mcaIndex  = info.get('McaIndex', mcaindex)
        fileIndex = info.get('FileIndex', fileindex)

        if fileIndex is None:
            if mcaIndex == 2:
                fileIndex = 0
            elif mcaIndex == 0:
                fileIndex = 1
            else:
                fileIndex = 0
                
        for i in range(3):
            if i not in [mcaIndex, fileIndex]:
                otherIndex = i
                break
        self.mcaIndex = mcaIndex
        self.fileIndex = fileIndex
        self.otherIndex = otherIndex
        
        self._stack.info['McaCalib'] = info.get('McaCalib', [0.0, 1.0, 0.0])
        self._stack.info['Channel0'] = info.get('Channel0', 0.0)
        self._stack.info['McaIndex'] = mcaIndex
        self._stack.info['FileIndex'] = fileIndex
        self._stack.info['OtherIndex'] = otherIndex
        self.stackUpdated()

    def stackUpdated(self):
        """
        Recalculates the different images associated to the stack
        """
        if isinstance(self._stack.data, numpy.ndarray):
            self._stackImageData = numpy.sum(self._stack.data, self.mcaIndex)
            #original ICR mca
            if DEBUG:
                print "(self.otherIndex, self.fileIndex) = ", (self.otherIndex, self.fileIndex)
            i = max(self.otherIndex, self.fileIndex)
            j = min(self.otherIndex, self.fileIndex)                
            mcaData0 = numpy.sum(numpy.sum(self._stack.data, i), j) * 1.0
        else:
            if DEBUG:
                t0 = time.time()
            shape = self._stack.data.shape
            if self.mcaIndex == 2:
                self._stackImageData = numpy.zeros((shape[0], shape[1]),
                                                self._stack.data.dtype)
                mcaData0 = numpy.zeros((shape[2],), numpy.float)
                step = 1
                for i in range(shape[0]):
                    tmpData = self._stack.data[i:i+step,:,:]
                    numpy.add(self._stackImageData[i:i+step,:],
                              numpy.sum(tmpData, 2),
                              self._stackImageData[i:i+step,:])
                    tmpData.shape = step*shape[1], shape[2]
                    numpy.add(mcaData0, numpy.sum(tmpData, 0), mcaData0)
            elif self.mcaIndex == 0:
                self._stackImageData = numpy.zeros((shape[1], shape[2]),
                                                self._stack.data.dtype)
                mcaData0 = numpy.zeros((shape[0],), numpy.float)
                step = 1
                for i in range(shape[0]):
                    tmpData = self._stack.data[i:i+step,:,:]
                    tmpData.shape = tmpData.shape[1:]
                    numpy.add(self._stackImageData,
                              tmpData,
                              self._stackImageData)
                    mcaData0[i] = tmpData.sum()
            if DEBUG:
                print "Print dynamic loading elapsed = ", time.time() -t0

        if DEBUG:
            print "__stackImageData.shape = ",  self._stackImageData.shape               
        calib = self._stack.info.get('McaCalib', [0.0, 1.0, 0.0])
        dataObject = DataObject.DataObject()
        dataObject.info = {"McaCalib": calib,
                           "selectiontype":"1D",
                           "SourceName":"Stack",
                           "Key":"SUM"}

        dataObject.x = [numpy.arange(len(mcaData0)).astype(numpy.float)
                        + self._stack.info.get('Channel0', 0.0)]

        dataObject.y = [mcaData0]

        #store the original spectrum
        self._mcaData0 = dataObject

        #add the original image
        self.showOriginalImage()

        #add the mca
        self.showOriginalMca()

        #calculate the ROIs
        self._ROIDict = {'name': "ICR",
                         'type': "CHANNEL",
                         'calibration':calib,
                         'from': dataObject.x[0][0],
                         'to': dataObject.x[0][-1]}
        
        self.updateROIImages()
        for key in self.pluginInstanceDict.keys():
            self.pluginInstanceDict[key].stackUpdated()

    def updateROIImages(self, ddict=None):
        if ddict is None:
            updateROIDict = False
            ddict = self._ROIDict
        else:
            updateROIDict = True
            
        xw =  ddict['calibration'][0] + \
              ddict['calibration'][1] * self._mcaData0.x[0] + \
              ddict['calibration'][2] * self._mcaData0.x[0] * \
                                        self._mcaData0.x[0]
        if ddict["name"] == "ICR":
            i1 = 0
            i2 = self._stack.data.shape[self.mcaIndex]
            imiddle = int(0.5 * (i1+i2))
            pos = 0.5 * (ddict['from'] + ddict['to'])
            imiddle = max(numpy.nonzero(xw <= pos)[0])
        elif ddict["type"].upper() != "CHANNEL":
            #energy like ROI
            if xw[0] < xw[-1]:
                i1 = numpy.nonzero(ddict['from'] <= xw)[0]
                if len(i1):
                    i1 = min(i1)
                else:
                    if DEBUG:
                        print "updateROIImages: nothing to be made"
                    return
                i2 = numpy.nonzero(xw <= ddict['to'])[0]
                if len(i2):
                    i2 = max(i2) + 1
                else:
                    if DEBUG:
                        print "updateROIImages: nothing to be made"
                    return
                pos = 0.5 * (ddict['from'] + ddict['to'])
                imiddle = max(numpy.nonzero(xw <= pos)[0])
            else:
                i2 = numpy.nonzero(ddict['from']<= xw)[0]
                if len(i2):
                    i2 = max(i2)
                else:
                    if DEBUG:
                        print "updateROIImages: nothing to be made"
                    return
                i1 = numpy.nonzero(xw <= ddict['to'])[0]
                if len(i1):
                    i1 = min(i1) + 1
                else:
                    if DEBUG:
                        print "updateROIImages: nothing to be made"
                    return
                pos = 0.5 * (ddict['from'] + ddict['to'])
                imiddle = min(numpy.nonzero(xw <= pos)[0])
        else:
            i1 = numpy.nonzero(ddict['from'] <= self._mcaData0.x[0])[0]
            if len(i1):
                i1 = min(i1)
            else:
                i1 = 0
            i1 = max(i1, 0)

            i2 = numpy.nonzero(self._mcaData0.x[0] <= ddict['to'])[0]
            if len(i2):
                i2 = max(i2)
            else:
                i2 = 0
            i2 = min(i2+1, self._stack.data.shape[self.mcaIndex])
            pos = 0.5 * (ddict['from'] + ddict['to'])
            imiddle = max(numpy.nonzero(self._mcaData0.x[0] <= pos)[0])
            xw = self._mcaData0.x[0]
            
        self._ROIImageDict = self.calculateROIImages(i1, i2, imiddle)
        if updateROIDict:
            self._ROIDict.update(ddict)

        roiKeys = ['ROI', 'Maximum', 'Minimum', 'Left', 'Middle', 'Right', 'Background']
        nImages = len(roiKeys)
        imageList  = [None] * nImages
        for i in xrange(nImages):
            key = roiKeys[i]
            imageList[i] = self._ROIImageDict[key]

        title = "%s" % ddict["name"]
        if ddict["name"] == "ICR":
            cursor = "Energy"
            if abs(ddict['calibration'][0]) < 1.0e-5:
                if abs(ddict['calibration'][1]-1) < 1.0e-5:
                    if abs(ddict['calibration'][2]) < 1.0e-5:
                        cursor = "Channel"
        elif ddict["type"].upper() == "CHANNEL":
            cursor = "Channel"
        else:
            cursor = ddict["type"]

        imageNames=[title,
                 '%s Maximum' % title,
                 '%s Minimum' % title,
                 '%s %.6g' % (cursor, xw[i1]),
                 '%s %.6g' % (cursor, xw[imiddle]),
                 '%s %.6g' % (cursor, xw[(i2-1)]),
                 '%s Background' %title]

        self.showROIImageList(imageList, image_names=imageNames)
                    
    def showOriginalImage(self):
        if DEBUG:
            print "showOriginalImage to be implemented" 

    def showOriginalMca(self):
        if DEBUG:
            print "showOriginalMca to be implemented" 

    def showROIImageList(self, imageList, image_names=None):
        self._ROIImageList  = imageList
        self._ROIImageNames = image_names
        self._stackROIImageListUpdated()

    def _stackROIImageListUpdated(self):
        for key in self.pluginInstanceDict.keys():
            self.pluginInstanceDict[key].stackROIImageListUpdated()

    def getStackROIImagesAndNames(self):
        return self._ROIImageList, self._ROIImageNames

    def calculateMcaDataObject(self, normalize=False):
        #original ICR mca
        if self._stackImageData is None:
            return
        mcaData = None
        if self._selectionMask is None:
            if normalize:
                npixels = self._stackImageData.shape[0] *\
                          self._stackImageData.shape[1] * 1.0
                dataObject = DataObject.DataObject()
                dataObject.info.update(self._mcaData0.info)
                dataObject.x  = [self._mcaData0.x[0]]
                dataObject.y =  [self._mcaData0.y[0] / npixels];
            else:
                dataObject = self._mcaData0
            return dataObject
        npixels = self._selectionMask.sum()
        if npixels == 0:
            if normalize:
                npixels = self._stackImageData.shape[0] * self._stackImageData.shape[1] * 1.0
                dataObject = DataObject.DataObject()
                dataObject.info.update(self._mcaData0.info)
                dataObject.x  = [self._mcaData0.x[0]]
                dataObject.y =  [self._mcaData0.y[0] / npixels];
            else:
                dataObject = self._mcaData0
            return dataObject

        mcaData = numpy.zeros(self._mcaData0.y[0].shape, numpy.float)

        n_nonselected = self._stackImageData.shape[0] *\
                        self._stackImageData.shape[1] - npixels
        if n_nonselected < npixels:
            arrayMask = (self._selectionMask == 0)
        else:
            arrayMask = (self._selectionMask > 0)
        cleanMask = numpy.nonzero(arrayMask)
        if DEBUG:
            print "self.fileIndex, self.mcaIndex", self.fileIndex, self.mcaIndex
        if DEBUG:
            t0 = time.time()            
        if len(cleanMask[0]) and len(cleanMask[1]):
            cleanMask = numpy.array(cleanMask).transpose()
            if self.fileIndex == 2:
                if self.mcaIndex == 0:
                    if isinstance(self.stack.data, numpy.ndarray):
                        for r, c in cleanMask:
                            mcaData += self.stack.data[:,r,c]
                    else:
                        if DEBUG:
                            print "Dynamic loading case 0"
                        #no other choice than to read all images
                        #for the time being, one by one
                        for i in xrange(self.stack.data.shape[0]):
                            tmpData = self.stack.data[i:i+1,:,:]
                            tmpData.shape = tmpData.shape[1:]
                            mcaData[i] = (tmpData*arrayMask).sum()
                elif self.mcaIndex == 1:
                    if isinstance(self.stack.data, numpy.ndarray):
                        for r, c in cleanMask:
                            mcaData += self.stack.data[r,:,c]
                    else:
                        raise IndexError, "Dynamic loading case 1"
                else:
                    raise IndexError, "Wrong combination of indices. Case 0"
            elif self.fileIndex == 1:
                if self.mcaIndex == 0:
                    if isinstance(self.stack.data, numpy.ndarray):
                        for r, c in cleanMask:
                            mcaData += self.stack.data[:,r,c]
                    else:
                        if DEBUG:
                            print "Dynamic loading case 2"
                        #no other choice than to read all images
                        #for the time being, one by one
                        for i in xrange(self.stack.data.shape[0]):
                            tmpData = self.stack.data[i:i+1,:,:]
                            tmpData.shape = tmpData.shape[1:]
                            #multiplication is faster than selection
                            #tmpData[arrayMask].sum() in my machine
                            mcaData[i] = (tmpData*arrayMask).sum()
                elif self.mcaIndex == 2:
                    if isinstance(self.stack.data, numpy.ndarray):
                        for r, c in cleanMask:
                            mcaData += self.stack.data[r,c,:]
                    else:
                        if DEBUG:
                            print "Dynamic loading case 3"
                        #try to minimize access to the file
                        row_dict = {}
                        for r, c in cleanMask:
                            if r not in row_dict.keys():
                                key = '%d' % r
                                row_dict[key] = []
                            row_dict[key].append(c)
                        for key in row_dict.keys():
                            r = int(key)
                            tmpMcaData = self.stack.data[r:r+1, row_dict[key],:]
                            tmpMcaData.shape = 1, -1
                            mcaData += numpy.sum(tmpMcaData,0)
                else:
                    raise IndexError, "Wrong combination of indices. Case 1"
            elif self.fileIndex == 0:
                if self.mcaIndex == 1:
                    if isinstance(self._stack.data, numpy.ndarray):
                        for r, c in cleanMask:
                            mcaData += self._stack.data[r,:,c]
                    else:
                        raise IndexError, "Dynamic loading case 4"
                elif self.mcaIndex == 2:
                    if isinstance(self._stack.data, numpy.ndarray):
                        for r, c in cleanMask:
                            mcaData += self._stack.data[r,c,:]
                    else:
                        if DEBUG:
                            print "Dynamic loading case 5"
                        #try to minimize access to the file
                        row_dict = {}
                        for r, c in cleanMask:
                            if r not in row_dict.keys():
                                key = '%d' % r
                                row_dict[key] = []
                            row_dict[key].append(c)
                        for key in row_dict.keys():
                            r = int(key)
                            tmpMcaData = self._stack.data[r:r+1, row_dict[key],:]
                            tmpMcaData.shape = 1, -1
                            mcaData += numpy.sum(tmpMcaData,0)
                else:
                    raise IndexError, "Wrong combination of indices. Case 2"
            else:
                raise IndexError, "File index undefined"
        if DEBUG:
            print "Mca sum elapsed = ", time.time() - t0
        if n_nonselected < npixels:
            mcaData = self._mcaData0.y[0] - mcaData

        if normalize:
            mcaData = mcaData/npixels

        calib = self._stack.info['McaCalib']
        dataObject = DataObject.DataObject()
        dataObject.info = {"McaCalib": calib,
                           "selectiontype":"1D",
                           "SourceName":"Stack",
                           "Key":"Selection"}
        dataObject.x = [numpy.arange(len(mcaData)).astype(numpy.float)
                        + self._stack.info['Channel0']]
        dataObject.y = [mcaData]

        return dataObject

    def calculateROIImages(self, index1, index2, imiddle=None):
        i1  = min(index1, index2)
        i2  = max(index1, index2)
        if imiddle is None:
            imiddle = int(0.5 * (i1 + i2))

        if i1 == i2:
            dummy = self._stackImageData * 0.0
            imageDict = {'ROI': dummy,
                      'Maximum': dummy,
                      'Minimum': dummy,
                      'Left': dummy,
                      'Middle': dummy,
                      'Right': dummy,
                      'Background':dummy}
            return imageDict

        if self.fileIndex == 0:
            if self.mcaIndex == 1:
                leftImage = self._stack.data[:,i1,:]
                middleImage = self._stack.data[:,imiddle,:]
                rightImage = self._stack.data[:,i2-1,:]
                dataImage = self._stack.data[:,i1:i2,:]
                maxImage = numpy.max(dataImage, 1)
                minImage = numpy.min(dataImage, 1)
                background =  0.5 * (i2-i1) * (leftImage+rightImage)                    
                roiImage = numpy.sum(dataImage,1)
            else:
                if DEBUG:
                    t0 = time.time()
                if isinstance(self._stack.data, numpy.ndarray):
                    leftImage = self._stack.data[:,:,i1]
                    middleImage = self._stack.data[:,:,imiddle]
                    rightImage = self._stack.data[:,:,i2-1]
                    dataImage = self._stack.data[:,:,i1:i2]
                    maxImage = numpy.max(dataImage, 2)
                    minImage = numpy.min(dataImage, 2)
                    background =  0.5 * (i2-i1) * (leftImage+rightImage)
                    roiImage = numpy.sum(dataImage,2)
                else:
                    shape = self._stack.data.shape
                    roiImage = self._stackImageData * 0
                    background = roiImage * 1
                    leftImage = roiImage * 1
                    middleImage = roiImage * 1
                    rightImage = roiImage * 1
                    maxImage = roiImage * 1
                    minImage = roiImage * 1
                    step = 1
                    for i in range(shape[0]):
                        tmpData = self._stack.data[i:i+step,:, i1:i2] * 1
                        numpy.add(roiImage[i:i+step,:],
                              numpy.sum(tmpData, 2),
                              roiImage[i:i+step,:])
                        numpy.add(minImage[i:i+step,:],
                                  numpy.min(tmpData, 2),
                                  minImage[i:i+step,:])
                        numpy.add(maxImage[i:i+step,:],
                                  numpy.max(tmpData, 2),
                                  maxImage[i:i+step,:])                            
                        leftImage[i:i+step, :]   += tmpData[:, :, 0]
                        middleImage[i:i+step, :] += tmpData[:, :, imiddle-i1]
                        rightImage[i:i+step, :]  += tmpData[:, :,-1]                                                   
                    background = 0.5*(i2-i1)*(leftImage+rightImage)
                if DEBUG:
                    print "ROI image calculation elapsed = ", time.time() - t0
        elif self.fileIndex == 1:
            if self.mcaIndex == 0:
                if isinstance(self._stack.data, numpy.ndarray):
                    leftImage  = self._stack.data[i1,:,:]
                    middleImage= self._stack.data[imiddle,:,:]
                    rightImage = self._stack.data[i2-1,:,:]
                    dataImage = self._stack.data[i1:i2,:,:]
                    maxImage = numpy.max(dataImage, 0)
                    minImage = numpy.min(dataImage, 0)
                    background =  0.5 * (i2-i1) * (leftImage+rightImage)
                    roiImage = numpy.sum(dataImage, 0)
                else:
                    shape = self._stack.data.shape
                    roiImage = self._stackImageData * 0
                    background = roiImage * 1
                    leftImage = roiImage * 1
                    middleImage = roiImage * 1
                    rightImage = roiImage * 1
                    maxImage = roiImage * 1
                    minImage = roiImage * 1
                    if DEBUG:
                        t0 = time.time()
                    istep = 1
                    for i in range(i1, i2):
                        tmpData = self._stack.data[i:i+istep,:,:]
                        tmpData.shape = roiImage.shape
                        numpy.add(roiImage,
                                  tmpData,
                                  roiImage)
                        if i == i1:
                            minImage = tmpData
                            maxImage = tmpData
                        else:
                            minMask  = tmpData < minImage
                            maxMask  = tmpData > minImage
                            minImage[minMask] = tmpData[minMask]
                            maxImage[maxMask] = tmpData[maxMask]
                        if (i == i1):
                            leftImage = tmpData
                        if (i == imiddle):
                            middleImage = tmpData                            
                        if i == (i2-1):
                            rightImage = tmpData
                    if DEBUG:
                        print "Dynamic ROI elapsed = ", time.time() - t0
                    if i2 > i1:
                        background = (leftImage + rightImage) * 0.5 * (i2-i1)
            else:
                if DEBUG:
                    t0 = time.time()
                if isinstance(self._stack.data, numpy.ndarray):
                    leftImage = self._stack.data[:,:,i1]
                    middleImage= self._stack.data[:,:,imiddle]
                    rightImage = self._stack.data[:,:,i2-1]
                    dataImage = self._stack.data[:,:,i1:i2]
                    maxImage = numpy.max(dataImage, 2)
                    minImage = numpy.min(dataImage, 2)
                    background =  0.5 * (i2-i1) * (leftImage+rightImage)
                    roiImage = numpy.sum(dataImage,2)
                else:
                    shape = self._stack.data.shape
                    roiImage = self._stackImageData * 0
                    background = roiImage * 1
                    leftImage = roiImage * 1
                    middleImage = roiImage * 1
                    rightImage = roiImage * 1
                    maxImage = roiImage * 1
                    minImage = roiImage * 1
                    step = 1
                    for i in range(shape[0]):
                        tmpData = self._stack.data[i:i+step,:, i1:i2] * 1
                        numpy.add(roiImage[i:i+step,:],
                              numpy.sum(tmpData, 2),
                              roiImage[i:i+step,:])
                        numpy.add(minImage[i:i+step,:],
                                  numpy.min(tmpData, 2),
                                  minImage[i:i+step,:])
                        numpy.add(maxImage[i:i+step,:],
                                  numpy.max(tmpData, 2),
                                  maxImage[i:i+step,:])                            
                        leftImage[i:i+step, :]   += tmpData[:, :, 0]
                        middleImage[i:i+step, :] += tmpData[:, :, imiddle-i1]
                        rightImage[i:i+step, :]  += tmpData[:, :,-1]                                                        
                    background = 0.5*(i2-i1)*(leftImage+rightImage)
                if DEBUG:
                    print "ROI image calculation elapsed = ", time.time() - t0
        else:
            #self.fileIndex = 2
            if self.mcaIndex == 0:
                leftImage = self._stack.data[i1,:,:]
                middleImage= self._stack.data[imiddle,:,:]
                rightImage = self._stack.data[i2-1,:,:]
                background =  0.5 * (i2-i1) * (leftImage+rightImage)
                dataImage = self._stack.data[i1:i2,:,:]
                minImage = numpy.min(dataImage, 0)
                maxImage = numpy.max(dataImage, 0)
                roiImage = numpy.sum(dataImage,0)
            else:
                leftImage = self._stack.data[:,i1,:]
                middleImage = self._stack.data[:,imidle,:]
                rightImage = self._stack.data[:,i2-1,:]
                background =  0.5 * (i2-i1) * (leftImage+rightImage)
                dataImage = self._stack.data[:,i1:i2,:]
                minImage = numpy.min(dataImage, 1)
                maxImage = numpy.max(dataImage, 1)
                roiImage = numpy.sum(dataImage,1)

        imageDict = {'ROI': roiImage,
                     'Maximum': maxImage,
                     'Minimum': minImage,
                     'Left': leftImage,
                     'Middle': middleImage,
                     'Right': rightImage,
                     'Background':background}

        return imageDict

    def setSelectionMask(self, mask):
        if DEBUG:
            print "setSelectionMask called"
        self._selectionMask = mask
        for key in self.pluginInstanceDict.keys():
            self.pluginInstanceDict[key].selectionMaskUpdated()

    def getSelectionMask(self):
        if DEBUG:
            print "getSelectionMask called"

    def addImage(self, image, name, info=None, replace=False, replot=True):
        """
        Add image data to the RGB correlator
        """
        print "Add image data not implemented"

    def removeImage(self, name, replace=True):
        """
        Remove image data from the RGB correlator
        """
        print "Remove image data not implemented"


    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True):
        """
        Add the 1D curve given by x an y to the graph.
        """
        print "addCurve not implemented"
        return None

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        """
        print "removeCurve not implemented"
        return None
    
    def getActiveCurve(self):
        """
        Function to access the currently active curve.
        It returns None in case of not having an active curve.

        Default output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.

        If just_legend is True:
            The legend of the active curve (or None) is returned.
        """
        if DEBUG:
            print "getActiveCurve default implementation"
        info = {}
        info['xlabel'] = 'Channel'
        info['ylabel'] = 'Counts'
        legend = 'ICR Spectrum'
        return self._mcaData0.x[0], self._mcaData0.y[0], legend , info

    def getGraphXLimits(self):
        if DEBUG:
            print "getGraphXLimits default implementation"
        return self._mcaData0.x[0].min(), self._mcaData0.x[0].max()

    def getGraphYLimits(self):
        if DEBUG:
            print "getGraphYLimits default implementation"
        return self._mcaData0.y[0].min(), self._mcaData0.y[0].max()

    def getStackDataObject(self):
        return self._stack

    def getStackData(self):
        return self._stack.data

    def getStackInfo(self):
        return self._stack.info
    
def test():
    #create a dummy stack
    nrows = 100
    ncols = 200
    nchannels = 1024
    a = numpy.ones((nrows, ncols), numpy.float)
    stackData = numpy.zeros((nrows, ncols, nchannels), numpy.float)
    for i in xrange(nchannels):
        stackData[:, :, i] = a * i
    stack = StackBase()
    
    stack.setStack(stackData, mcaindex=2)
    print "This should be 0 = ",  stack.calculateROIImages(0, 0)['ROI'].sum()
    print "This should be 0 = ",  stack.calculateROIImages(0, 1)['ROI'].sum()
    print stackData[:,:,0:10].sum(), "should be =", stack.calculateROIImages(0, 10)['ROI'].sum()

if __name__ == "__main__":
    test()
    

#/*##########################################################################
# Copyright (C) 2008-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it 
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
import os
import sys
import re
import struct
import numpy
from PyMca import DataObject

DEBUG = 0
SOURCE_TYPE="EdfFileStack"

class OmnicMap(DataObject.DataObject):
    def __init__(self, filename):
        DataObject.DataObject.__init__(self)
        fileSize = os.path.getsize(filename)
        if sys.platform == 'win32':
            fid = open(filename, 'rb')
        else:
            fid = open(filename, 'r')
        data=fid.read()
        fid.close()

        try:
            omnicInfo = self.getOmnicInfo(data)
        except:
            omnicInfo = None
        self.sourceName = [filename]
        if sys.version < '3.0':
            searchedChain = "Spectrum "
        else:
            searchedChain = bytes("Spectrum ", 'utf-8')
        firstByte = data.index(searchedChain)
        s = data[firstByte:(firstByte+100-16)]
        if sys.version >= '3.0':
            s = str(s)
        if DEBUG:
            print("firstByte = %d" % firstByte)
            print("s1 = %s " % s)
        exp = re.compile('(-?[0-9]+\.?[0-9]*)')
        tmpValues= exp.findall(s)
        spectrumIndex = int(tmpValues[0])
        self.nSpectra = int(tmpValues[1])
        if "X = " in s:
            xPosition = float(tmpValues[2])
            yPosition = float(tmpValues[3])
        else:
            #I have to calculate them from the scan
            xPosition, yPosition = self.getPositionFromIndexAndInfo(0, omnicInfo)
        if DEBUG:
            print("spectrumIndex, nSpectra, xPosition, yPosition = %d %d %f %f" %\
                    (spectrumIndex, self.nSpectra, xPosition, yPosition))
        if sys.version < '3.0':
            chain = "Spectrum"
        else:
            chain = bytes("Spectrum", 'utf-8')
        secondByte = data[(firstByte+1):].index(chain)
        secondByte += firstByte + 1
        if DEBUG:
            print("secondByte = ", secondByte)
        self.nChannels = int((secondByte - firstByte - 100)/4)
        if DEBUG:
            print("nChannels = %d" % self.nChannels)
        self.firstSpectrumOffset = firstByte - 16

        #fill the header
        self.header =[]
        oldXPosition = xPosition
        oldYPosition = yPosition
        self.nRows = 0
        for i in range(self.nSpectra):
            offset = int(firstByte + i * (100 + self.nChannels * 4))
            if sys.version < '3.0':
                s = data[offset:(offset+100-16)]
            else:
                s = str(data[offset:(offset+100-16)])
            tmpValues= exp.findall(s)
            spectrumIndex = int(tmpValues[0])
            self.nSpectra = int(tmpValues[1])
            if "X = " in s:
                xPosition = float(tmpValues[2])
                yPosition = float(tmpValues[3])
            else:
                #I have to calculate them from the scan
                xPosition, yPosition = self.getPositionFromIndexAndInfo(i, omnicInfo)                
            if (abs(yPosition-oldYPosition)>1.0e-6) and\
               (abs(xPosition-oldXPosition)<1.0e-6):
                break
            self.nRows = self.nRows + 1
        if DEBUG:
            print("DIMENSIONS X = %f Y=%d" %\
                  ((self.nSpectra*1.0)/self.nRows ,self.nRows))
        
        #arrange as an EDF Stack
        self.info = {}
        self.__nFiles = int((self.nSpectra)/self.nRows)
        self.data = numpy.zeros((self.__nFiles,
                                 self.nRows,
                                 self.nChannels),
                                 numpy.float32)

        self.__nImagesPerFile = 1
        offset = firstByte - 16 + 100 # starting position of the data
        delta = 100 + self.nChannels * 4
        fmt = "%df" % self.nChannels
        for i in range(self.__nFiles):
            for j in range(self.nRows):
                self.data[i,j,:] = struct.unpack(fmt,\
                                        data[offset:(offset+delta-100)])
                offset = int(offset + delta)
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i+1,)
            self.info[key] = shape[i]

        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = self.sourceName
        self.info["Size"]       = self.__nFiles * self.__nImagesPerFile
        self.info["NumberOfFiles"] = self.__nFiles * 1
        self.info["FileIndex"] = 0
        self.info["Channel0"] = 0.0
        if omnicInfo is not None:
            self.info['McaCalib'] = [omnicInfo['First X value']*1.0,
                                     omnicInfo['Data spacing']*1.0,
                                     0.0]
        else:
            self.info["McaCalib"] = [0.0, 1.0, 0.0]

    def getOmnicInfo(self, data):
        #additional information
        fmt="I" #unsigned long in 32-bit
        offset=372 #93*4 unsigned integers
        infoBlockIndex = (struct.unpack(fmt,data[offset:(offset+4)])[0]-204)/4.
        infoBlockIndex = int(infoBlockIndex)
        #infoblock is the position of the information block
        offset = infoBlockIndex * 4
        #read 13 unsigned integers
        nValues = 13
        fmt = "%dI" % nValues
        values = struct.unpack(fmt,data[offset:(offset+4*nValues)])
        ddict = {}
        ddict['Number of points'] = values[0]
        ddict['Number of scan points'] = values[6] 
        ddict['Interferogram peak position'] = values[7]
        ddict['Number of sample scans'] = values[8]
        ddict['Number of FFT points'] = values[10]
        ddict['Number of background scans'] = values[12]
        offset = (infoBlockIndex + 3) * 4
        nFloats = 47
        fmt = "%df" % nFloats
        vFloats = struct.unpack(fmt,data[offset:(offset+4*nFloats)])
        lastX  = vFloats[0]
        firstX = vFloats[1]
        startIndicesOfSpectra = vFloats[14]
        laserFrequency = vFloats[16]
        ddict['First X value'] = firstX
        ddict['Last X value']  = lastX
        ddict['Identifier for start indices of spectra'] = vFloats[14]
        ddict['Laser frequency']  = vFloats[16]
        ddict['Data spacing']  = (lastX - firstX)/(ddict['Number of points']-1.0)
        ddict['Background gain'] = vFloats[10]
        if DEBUG:
            for key in ddict.keys():
                print(key, ddict[key])
        ddict.update(self.getMapInformation(data))
        return ddict

    def getMapInformation(self, data):
        #look for the chain 'Position'
        if sys.version < '3.0':
            chain = 'Position'
        else:
            chain = bytes('Position', 'utf-8')
        offset = data.index(chain)
        positions = [offset]
        while True:
            try:
                a = data[(offset+1):].index(chain)
                offset = a + offset + 1
                positions.append(offset)
            except ValueError:
                break

        ddict = {}
        #map description position
        if (positions[1] - positions[0]) == 66: #reverse engineered magic number :-)
            mapDescriptionOffset = positions[0] - 90
            mapDescription = struct.unpack('6f', data[mapDescriptionOffset:mapDescriptionOffset+24])
            y0, y1, deltaY, x0, x1, deltaX = mapDescription
            ddict['First map location'] = [x0, y0]
            ddict['Last map location'] = [x1, y1]
            ddict['Mapping stage X step size'] = deltaX
            ddict['Mapping stage Y step size'] = deltaY
            ddict['Number of spectra'] = abs((1+((y1-y0)/deltaY)) * (1+((x1 - x0) /deltaX)))
        if DEBUG:
            for key in ddict.keys():
                print(key, ddict[key])
        return ddict
            
    def getPositionFromIndexAndInfo(self, index, info=None):
        if info is None:
            return 0.0, 0.0
        ddict= info
        #first variation on X and then on Y
        try:
            x0, y0 = ddict['First map location']
        except KeyError:
            return 0.0, 0.0
        x1, y1 = ddict['Last map location']
        deltaX = ddict['Mapping stage X step size']
        deltaY = ddict['Mapping stage Y step size']
        nX = int(1+((x1 - x0) /deltaX))
        x = x0 + (index % nX) * deltaX
        y = y0 + int(index/nX) * deltaY
        return x, y

        
        
if __name__ == "__main__":
    filename = None
    if len(sys.argv) > 1:
        DEBUG = 1
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    elif os.path.exists("SambaPhg_IR.map"):
        filename = "SambaPhg_IR.map"
    if filename is not None:
        w = OmnicMap(filename)
        print(type(w))
        print(type(w.data[0:10]))
        print("shape = ", w.data.shape)
        print(type(w.info))
    else:
        print("Please supply input filename")

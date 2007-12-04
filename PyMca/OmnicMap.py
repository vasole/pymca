import os
import sys
import re
import struct
import DataObject
import numpy

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
        self.sourceName = [filename]        
        firstByte = data.index("Spectrum position")
        s = data[firstByte:(firstByte+100-16)]
        exp = re.compile('(-?[0-9]+\.?[0-9]*)')
        spectrumIndex, nSpectra, xPosition, yPosition = exp.findall(s)[0:4]
        spectrumIndex = int(spectrumIndex)
        self.nSpectra = int(nSpectra)
        xPosition = float(xPosition)
        yPosition = float(yPosition)
        if DEBUG:
            print "spectrumIndex, nSpectra, xPosition, yPosition = ", \
                spectrumIndex, nSpectra, xPosition, yPosition                
        secondByte = data.index("Spectrum position %d" % (spectrumIndex+1))
        self.nChannels = (secondByte - firstByte - 100)/4
        if DEBUG:
            print "nChannels = ", self.nChannels
        self.firstSpectrumOffset = firstByte - 16

        #fill the header
        self.header =[]
        oldXPosition = xPosition
        oldYPosition = yPosition
        self.nRows = 0
        for i in range(self.nSpectra):
            offset = firstByte + i * (100 + self.nChannels * 4)
            s = data[offset:(offset+100-16)]
            #print "s = ",s
            spectrumIndex, nSpectra, xPosition, yPosition = exp.findall(s)[0:4]
            xPosition = float(xPosition)
            yPosition = float(yPosition)
            if yPosition != oldYPosition:
                break
            self.nRows = self.nRows + 1
        if DEBUG:
            print "DIMENSIONS X = %f Y=%d" % ((self.nSpectra*1.0)/self.nRows ,self.nRows)
        
        #arrange as an EDF Stack
        self.info = {}
        self.__nFiles = (self.nSpectra)/self.nRows
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
                self.data[i,j,:] = struct.unpack(fmt, data[offset:(offset+delta-100)])
                offset = offset + delta
                
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i+1,)
            self.info[key] = shape[i]

        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = self.sourceName
        self.info["Size"]       = self.__nFiles * self.__nImagesPerFile
        self.info["NumberOfFiles"] = self.__nFiles * 1
        self.info["FileIndex"] = 0
        self.info["McaCalib"] = [0.0, 1.0, 0.0]
        self.info["Channel0"] = 0.0
        
if __name__ == "__main__":
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    elif os.path.exists("SambaPhg_IR.map"):
        filename = "SambaPhg_IR.map"
    if filename is not None:
        DEBUG = 1   
        w = OmnicMap(filename)
    else:
        print "Please supply input filename"

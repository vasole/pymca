import os
import sys
import struct
import DataObject
import numpy
import PyMcaIOHelper

DEBUG = 0
SOURCE_TYPE="EdfFileStack"

import time
class SupaVisioMap(DataObject.DataObject):
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
        e0 = time.time()

        values = struct.unpack("%dH" % (len(data)/2), data)
        data = numpy.array(values, numpy.uint16)
        #print values
        nrows = values[1]
        ncols = values[2]
        self.nSpectra = nrows * ncols
        data.shape = [len(data)/3, 3]
        self.nChannels = data[:,2].max() + 1

        #fill the header
        self.header =[]
        self.nRows = nrows

        #arrange as an EDF Stack
        self.info = {}
        self.__nFiles = (self.nSpectra)/self.nRows
        self.__nImagesPerFile = 1

        e0 = time.time()
        self.data = PyMcaIOHelper.fillSupaVisio(data);
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
    elif os.path.exists(".\PIGE\010826.pige"):
        filename = ".\PIGE\010826.pige"
    if filename is not None:
        DEBUG = 1   
        w = SupaVisioMap(filename)
    else:
        print "Please supply input filename"

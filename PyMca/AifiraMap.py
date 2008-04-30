import os
import sys
import struct
import DataObject
import numpy
import PyMcaIOHelper

DEBUG = 0
SOURCE_TYPE="EdfFileStack"

import time
class AifiraMap(DataObject.DataObject):
    def __init__(self, filename):
        DataObject.DataObject.__init__(self)
        
        fileSize = os.path.getsize(filename)
        if sys.platform == 'win32':
            fid = open(filename, 'rb')
        else:
            fid = open(filename, 'r')

        self.sourceName = [filename]
        
        self.data = PyMcaIOHelper.readAifira(fid).astype(numpy.float);

        nrows, ncols, nChannels = self.data.shape
        self.nSpectra = nrows * ncols

        fid.close()

        #fill the header
        self.header =[]
        self.nRows = nrows

        #arrange as an EDF Stack
        self.info = {}
        self.__nFiles = (self.nSpectra)/self.nRows
        self.__nImagesPerFile = 1

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
    elif os.path.exists("./AIFIRA/010737.DAT"):
        filename = "./AIFIRA/010737.DAT"
    if filename is not None:
        DEBUG = 1   
        w = AifiraMap(filename)
    else:
        print "Please supply input filename"

#/*##########################################################################
# Copyright (C) 2008-2014 European Synchrotron Radiation Facility
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
import copy
from PyMca import DataObject

if sys.version < '2.6':
    def bytes(x):
        return str(x)

DEBUG = 0
SOURCE_TYPE = "EdfFileStack"

class MRCMap(DataObject.DataObject):
    '''
    Class to read MRC files

    It reads the spectra into a DataObject instance.
    This class info member contains all the parsed information.
    This class data member contains the map itself as a 3D array.
    '''
    def __init__(self, filename):
        '''
        Parameters:
        -----------
        filename : str
            Name of the input file.
            It is expected to work with files generated at BESSY with raw2mrc
        '''
        DataObject.DataObject.__init__(self)
        try:
            fid = open(filename, 'rb')
            header = fid.read(1024)
            fid.close()
        except:
            fid.close()
            raise

        if not _isMRCHeader(header):
            raise IOError("File does not seem to be an MRC file")

        self.sourceName = filename

        # check endianness
        flag = struct.unpack("4B", header[212:216])[0]
        if flag in [17, 18]:
            # high endian (Motorola)
            endianness = ">"
        else:
            # little endian (Intel)
            endianness = "<"

        fmt = endianness + "10i"
        tenIntegers = struct.unpack(fmt, header[0:40])
        nColumns, nRows, nImages, mode = tenIntegers[0:4]


        fmt = endianness + "6f"
        sixFloats = struct.unpack(fmt, header[40:64])
        fmt = endianness + "3i"
        threeIntegers = struct.unpack(fmt, header[64:76])
        fmt = endianness + "3f"
        threeFloats = struct.unpack(fmt, header[76:88])

        # number of bytes in extended header
        fmt = endianness + "i"
        offset = struct.unpack(fmt, header[92:96])[0]

        fmt = endianness + "ii"
        imodStamp, imodFlags = struct.unpack(fmt, header[152:160])

        if mode == 0:
            # bytes
            dataFormat = endianness + "%dB" % (nRows * nColumns)
        elif mode == 1:
            # signed short integers (16 bit)
            dataFormat = endianness + "%dh" % (nRows * nColumns)
        elif mode == 2:
            # float
            dataFormat = endianness + "%df" % (nRows * nColumns)
        elif mode == 3:
            # two shorts, complex data
            dataFormat = endianness + "%dh" % (2 * nRows * nColumns)
        elif mode == 4:
            # two floats, complex data
            dataFormat = endianness + "%df" % (2 * nRows * nColumns)
        elif mode == 6:
            # unsigned 16 bit integers (non-standard)
            dataFormat = endianness + "%dH" % (nRows * nColumns)
        elif mode == 16:
            # unsigned char * 3(rgb data, non-standard)
            dataFormat = endianness + "%dB" % (3 * nRows * nColumns)
        else:
            raise IOError("Data format not undestood")

        if imodFlags == 1:
            # bytes stored as signed
            dataFormat = dataFormat.lower()


        data = numpy.zeros((nImages, nRows * nColumns), numpy.float)
        fid = open(filename, 'rb')
        fileOffset = 1024 + offset
        fid.seek(fileOffset)
        dataSize= struct.calcsize(dataFormat)
        try:
            for i in range(nImages):
                tmpData = fid.read(dataSize)
                data[i] = struct.unpack(dataFormat, tmpData)
            fid.close()
        except:
            fid.close()
            raise

        data.shape = nImages, nRows, nColumns
        self.data = data

        self.info = {}
        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = self.sourceName
        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i + 1,)
            self.info[key] = shape[i]
        self.info["NumberOfFiles"] = 1
        self.info["McaIndex"] = 0
        self.info["McaCalib"] = [0.0, 1.0, 0.0]
        self.info["Channel0"] = 0.0

def _isMRCHeader(header):
    if sys.version < '3.0':
        test = "MAP "
    else:
        test = bytes("MAP ", "utf-8")
    if struct.unpack("4s", header[208:212])[0] == test:
        return True
    else:
        return False

def isMRCFile(filename):
    try:
        fid = open(filename, 'rb')
        header = fid.read(1024)
        fid.close()
    except:
        fid.close()
        return False
    nColumns, nRows, nImages = struct.unpack("iii", header[0:12])
    imodStamp, imodFlags = struct.unpack("ii", header[152:160])
    #print(imodStamp, imodFlags)

    # system byte order
    #print("system ",sys.byteorder)

    return _isMRCHeader(header)

if __name__ == "__main__":
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    print("is MRC File?", isMRCFile(filename))
    instance = MRCMap(filename)
    print(instance.info)
    print(instance.data)

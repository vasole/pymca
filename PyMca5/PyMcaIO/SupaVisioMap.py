#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
import sys
import os
import numpy
import struct
import time
from PyMca5 import DataObject
from PyMca5.PyMcaIO import PyMcaIOHelper

DEBUG = 0
SOURCE_TYPE="EdfFileStack"

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
        self.data = PyMcaIOHelper.fillSupaVisio(data).astype(numpy.float);
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
        print("Please supply input filename")

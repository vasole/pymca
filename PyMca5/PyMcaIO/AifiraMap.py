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
from PyMca5.PyMcaCore import DataObject
from PyMca5.PyMcaIO import PyMcaIOHelper

DEBUG = 0
SOURCE_TYPE = "EdfFileStack"


class AifiraMap(DataObject.DataObject):
    def __init__(self, filename):
        DataObject.DataObject.__init__(self)

        if sys.platform == 'win32':
            fid = open(filename, 'rb')
        else:
            fid = open(filename, 'r')

        self.sourceName = [filename]

        self.data = PyMcaIOHelper.readAifira(fid).astype(numpy.float)

        nrows, ncols, nChannels = self.data.shape
        self.nSpectra = nrows * ncols

        fid.close()

        #fill the header
        self.header = []
        self.nRows = nrows

        #arrange as an EDF Stack
        self.info = {}
        self.__nFiles = self.nSpectra / self.nRows
        self.__nImagesPerFile = 1

        shape = self.data.shape
        for i in range(len(shape)):
            key = 'Dim_%d' % (i + 1,)
            self.info[key] = shape[i]

        self.info["SourceType"] = SOURCE_TYPE
        self.info["SourceName"] = self.sourceName
        self.info["Size"] = self.__nFiles * self.__nImagesPerFile
        self.info["NumberOfFiles"] = self.__nFiles * 1
        self.info["FileIndex"] = 0
        self.info["McaCalib"] = [0.0, 1.0, 0.0]
        self.info["Channel0"] = 0.0


def main():
    global DEBUG

    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    elif os.path.exists("./AIFIRA/010737.DAT"):
        filename = "./AIFIRA/010737.DAT"
    if filename is not None:
        DEBUG = 1
        AifiraMap(filename)
    else:
        print("Please supply input filename")

if __name__ == "__main__":
    main()

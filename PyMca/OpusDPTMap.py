#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Software group"
import numpy
from PyMca import DataObject
from PyMca import specfilewrapper as specfile

SOURCE_TYPE = "SpecFileStack"


class OpusDPTMap(DataObject.DataObject):
    def __init__(self, filename):
        DataObject.DataObject.__init__(self)
        sf = specfile.Specfile(filename)
        scan = sf[1]
        data = scan.data()
        nMca, nchannels = data.shape
        nMca = nMca - 1
        xValues = data[0, :] * 1
        xValues.shape = -1
        if 0:
            self.data = numpy.zeros((nMca, nchannels), numpy.float32)
            self.data[:, :] = data[1:, :]
            self.data.shape = 1, nMca, nchannels
        else:
            self.data = data[1:, :]
            self.data.shape = 1, nMca, nchannels
        data = None

        #perform a least squares adjustment to a line
        x = numpy.arange(nchannels).astype(numpy.float32)
        Sxy = numpy.dot(x, xValues.T)
        Sxx = numpy.dot(x, x.T)
        Sx  = x.sum()
        Sy  = xValues.sum()
        d = nchannels * Sxx - Sx * Sx
        zero = (Sxx * Sy - Sx * Sxy) / d
        gain = (nchannels * Sxy - Sx * Sy) / d

        #and fill the requested information to be identified as a stack
        self.info['SourceName'] = [filename]
        self.info["SourceType"] = "SpecFileStack"
        self.info["Size"]       = 1, nMca, nchannels
        self.info["NumberOfFiles"] = 1
        self.info["FileIndex"] = 0
        self.info["McaCalib"] = [zero, gain, 0.0]
        self.info["Channel0"] = 0.0


def main():
    import sys
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if filename is not None:
        OpusDPTMap(filename)
    else:
        print("Please supply input filename")

if __name__ == "__main__":
    main()

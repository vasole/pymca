#!/usr/bin/env python
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
import os
import sys
import re
import DataObject
import numpy

DEBUG = 0
SOURCE_TYPE="EdfFileStack"

class LuciaMap(DataObject.DataObject):
    def __init__(self, filename, infofile=None):
        DataObject.DataObject.__init__(self)
        
        fileSize = os.path.getsize(filename)
        fid = open(filename, 'r')
        data=fid.read()
        fid.close()
        data.replace("\r\n","\n")
        self.sourceName = [filename]
        firstByte = data.index("\n\n")


        header = data[0:firstByte]
        #get rid of the date
        data = data[firstByte:]

        #leave only the '----' as separator
        data.replace("\r","")
        data.replace("\n","")
        sep = '-'
        while sep in data:
            sep = sep + '-'
        sep = sep[1:]
        data = data.split(sep)
        if len(data[0]) != len(data[-1]):
            if len(data[0]) > 1:
                del data[-1]
            else:
                del data[0]
                
        #get the number of channels
        exp = re.compile('(-?[0-9]+\.?[0-9]*)')
        spectrum = [float(x) for x in exp.findall(data[0])]
        self.nChannels = len(spectrum)
        self.nSpectra = len(data)
        self.nRows = self.nSpectra

        #try to get the information
        if infofile is None:
            infofile = ""
            split = filename.split('_')
            if len(split) > 1:
                for i in range(len(split)-1):
                    if i == 0:
                        infofile = split[i]
                    else:
                        infofile += "_" + split[i]
                infofile = infofile + "_Infos_" + split[-1].replace('.mca', '.dat') 
            
        if os.path.exists(infofile):
            info = self._getInfo(infofile)
            if info.has_key('vwidth') and info.has_key('vstep'):
                vwidth = info['vwidth']
                vstep = info['vstep']
                if abs(vstep) > 0:
                    self.nRows = int((vwidth / vstep) + 1)

        #fill the header
        self.header =header

        #arrange as an EDF Stack
        self.info = {}
        self.__nFiles = 1

        self.__nImagesPerFile = 1

        #self.nRows = 41
        self.nCols = self.nSpectra/self.nRows
        
        self.data = numpy.zeros((self.nRows,
                                 self.nCols,
                                 self.nChannels),
                                 numpy.float32)
        n = 0
        for i in range(self.nRows):
            for j in range(self.nCols):
                s = data[n]
                spectrum = numpy.array([float(x) for x in exp.findall(s)])
                self.data[i,j,:] = spectrum[:]
                n = n + 1

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

    def _getInfo(self, filename):
        # This dictionnary is to be internally normalized
        # for the time being no I0 nor dead time
        exp = re.compile('(-?[0-9]+\.?[0-9]*)')
        f = open(filename)
        data = f.readlines()
        f.close()
        ddict = {}
        for line in data:
            if line.startswith("# Horizontal center position"):
                ddict['center'] = [float(x) for x in exp.findall(line)][0]
            elif line.startswith("# Horizontal width"):
                ddict['hwidth'] =  [float(x) for x in exp.findall(line)][0]
            elif line.startswith("# Horizontal step"):
                ddict['hstep'] =  [float(x) for x in exp.findall(line)][0]
            elif line.startswith("# Vertical width"):
                ddict['vwidth'] =  [float(x) for x in exp.findall(line)][0]
            elif line.startswith("# Vertical step"):
                ddict['vstep'] =  [float(x) for x in exp.findall(line)][0]
        return ddict        
        
        
        
if __name__ == "__main__":
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    elif os.path.exists("S10S_6_01.mca"):
        filename = "S10S_6_01.mca"
    if filename is not None:
        DEBUG = 1   
        w = LuciaMap(filename)
    else:
        print("Please supply input filename")

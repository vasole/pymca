#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2008 European Synchrotron Radiation Facility
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
This clases just put in evidence the Specfile methods called from
PyMca.
It can be used to wrap other formats as specile
"""
import sys
import os
import numpy
import SpecFileAbstractClass

#spx file format is based on XML
import xml.sax
from xml.sax.handler import ContentHandler
import types

class SpectrumReader(ContentHandler):
    def __init__(self):
        ContentHandler.__init__(self)

        self._spectrum_list = []
        self.__last_spectrum = None
        self._axes = None
        self.__parsed_keys = ['RealTime', 'LifeTime', 'DeadTime',
                              'CalibAbs', 'CalibLin',
                              'HighVoltage', 'TubeCurrent']

        
    def spectrum_list(self):
        return self._spectrum_list


    def startElement(self, name, attributes):
        if name == 'ClassInstance':
            if attributes['Type'] == 'TRTSpectrum':
                # 'str' is used to decode from Unicode to normal ASCII
                self.__last_spectrum = { 'name': str(attributes['Name']) }
                self.__last_spectrum['CalibAbs'] = 0.0
                self.__last_spectrum['CalibLin'] = 1.0
                self._spectrum_list.append(self.__last_spectrum)
        elif name == 'AxesParameter' and self.__last_spectrum is not None:
                self._axes = []
                self.__last_spectrum['axes'] = self._axes
        elif name.startswith('Axis') and self._axes is not None:
            pos = attributes['AxisPosition'].replace(',', '.')
                        
            self._axes.append({ 'name': str(attributes['AxisName']),
                                 'pos': float(pos),
                                 'unit': str(attributes['AxisUnit']) })
        elif name == 'Channels' and self.__last_spectrum is not None:
            self.__last_spectrum['channels']=True

        self.__last_used_name = str(name)


    def characters(self, content):
        if self.__last_spectrum is None:
            return

        if self.__last_used_name.upper() == "CHANNELS":
            add_channels = self.__last_spectrum.get('channels', False)

            if add_channels and type(add_channels) != types.ListType:
                self.__last_spectrum['channels']=[int(c) for c in str(content).split(",")]
            
        elif  self.__last_used_name in ['Date', 'Time']:
            self.__last_spectrum[self.__last_used_name] = str(content)
        elif  self.__last_used_name in self.__parsed_keys:
            self.__last_spectrum[self.__last_used_name] = float(str(content).replace(',','.'))
        self.__last_used_name = ""

    def endElement(self, name):
        pass


class SPXFileParser(SpecFileAbstractClass.SpecFileAbstractClass, SpectrumReader):
    def __init__(self, filename):
        SpectrumReader.__init__(self)
        SpecFileAbstractClass.SpecFileAbstractClass.__init__(self, filename)
        if not os.path.exists(filename):
            raise IOError, "File %s does not exists"  % filename
        f = open(filename)
        info = f.readlines()
        f.close()
        if info[0].startswith('<?'):
            del info[0]
        info = "".join(info)
        xml.sax.parseString(info, self)
        info = self.spectrum_list()
        # I've got all the spectrum information
        # I have to leave it available in a way that
        # looks as a specfile scan

        #for the time being I consider only one spectrum
        info = info[0]
        data = numpy.array(info['channels'], numpy.float)
        data.shape =len(data), 1
        scanheader = ['#S 1  ' + info.get('name', "Unknown name")] 
        axes = info.get('axes', [])
        i = 0
        for axis in axes:
            scanheader.append("#U%d %s  %f  %s" % (i,
                                             axis['name'],
                                             axis['pos'],
                                             axis['unit']))
            i += 1
        for key in ['HighVoltage', 'TubeCurrent']:
            scanheader.append("#U%d %s %s" % (i, key, info.get(key, "Unknown")))
            i += 1

        scanheader.append("#@CALIB %f %f 0" % (info.get('CalibAbs', 0.0),
                                               info.get('CalibLin', 1.0)))
                                               
            
        self.scandata = [SpecFileAbstractClass.SpecFileAbstractScan(data,
                                scantype="MCA",
                                scanheader=scanheader)]

def test(filename):
    SPXFileParser(filename)

if __name__ == "__main__":
    test(sys.argv[1])
        

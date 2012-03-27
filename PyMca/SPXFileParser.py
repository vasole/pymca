#!/usr/bin/env python
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
import sys
import os
import numpy
import types

#spx file format is based on XML
import xml.etree.ElementTree as ElementTree
from PyMca import SpecFileAbstractClass

def myFloat(x):
    try:
        return float(x)
    except ValueError:
        if ',' in x:
            try:
                return float(x.replace(',','.'))
            except:
                return float(x)
        elif '.' in x:
            try:
                return float(x.replace('.',','))
            except:
                return float(x)
        else:
            raise

class SPXFileParser(SpecFileAbstractClass.SpecFileAbstractClass):
    def __init__(self, filename):
        SpecFileAbstractClass.SpecFileAbstractClass.__init__(self, filename)
        if not os.path.exists(filename):
            raise IOError("File %s does not exists"  % filename)
        f = ElementTree.parse(filename)
        root = f.getroot()
        info = {}
        #I do not use find all, providing support for only one spectrum
        infoKeys = ['HighVoltage', 'TubeCurrent',
                    'RealTime', 'LifeTime', 'DeadTime',
                    'ZeroPeakPosition', 'ZeroPeakFrequency', 'PulseDensity',
                    'Amplification', 'ShapingTime',
                    'Date','Time',
                    'ChannelCount','CalibAbs', 'CalibLin']

        for key in infoKeys:
            keyToSearch = './/ClassInstance/%s' % key
            content = root.find(keyToSearch)
            if content is not None:
                info[key] = content.text
        axes = root.find('.//ClassInstance/AxesParameter')
        data = numpy.array([float(x) for x in root.find('.//Channels').text.split(',')])
        data.shape = len(data), 1

        scanheader = ['#S 1  ' + info.get('name', "Unknown name")] 
        i = 0
        if axes is not None:
            for axis in axes:
                scanheader.append("#U%d %s  %f  %s" % (i,
                                                 axis.attrib['AxisName'],
                                                 myFloat(axis.attrib['AxisPosition']),
                                                 axis.attrib['AxisUnit']))
                i += 1
        for key in infoKeys:
            scanheader.append("#U%d %s %s" % (i, key, info.get(key, "Unknown")))
            i += 1

        scanheader.append("#@CALIB %f %f 0" % (myFloat(info.get('CalibAbs', 0.0)),
                                               myFloat(info.get('CalibLin', 1.0))))
        
        self.scandata = [SpecFileAbstractClass.SpecFileAbstractScan(data,
                                scantype="MCA",
                                scanheader=scanheader)]

def test(filename):
    SPXFileParser(filename)

if __name__ == "__main__":
    test(sys.argv[1])
        

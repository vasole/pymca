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
import numpy
import copy

class DataObject(object):
    def __init__(self):
        self.info = {}
        self.data = numpy.array([])

    def getInfo(self):
        return self.info
    
    def getData(self):
        return self.data 

    if 0:
        def select(self,selection=None):
            if selection is None:
                return copy.deepcopy(self.data)
            else:
                print("Not implemented (yet)")
                #it will be a new array
                return copy.deepcopy(self.data)
    else:
        def select(self,selection=None):            
            dataObject = DataObject()
            dataObject.info = self.info
            dataObject.info['selection'] = selection
            if selection is None:
                dataObject.data = self.data 
                return dataObject
            if type(selection) == type({}):
                #dataObject.data = self.data #should I set it to none???    
                dataObject.data = None
                if 'rows' in selection:
                    dataObject.x = None
                    dataObject.y = None
                    dataObject.m = None
                    if 'x' in selection['rows']:
                        for rownumber in selection['rows']['x']:
                            if rownumber is None:continue
                            if dataObject.x is None:dataObject.x = []
                            dataObject.x.append(self.data[rownumber,:])
                
                    if 'y' in selection['rows']:
                        for rownumber in selection['rows']['y']:
                            if rownumber is None:continue
                            if dataObject.y is None:dataObject.y = []
                            dataObject.y.append(self.data[rownumber,:])

                    if 'm' in selection['rows']:
                        for rownumber in selection['rows']['m']:
                            if rownumber is None:continue
                            if dataObject.m is None:dataObject.m = []
                            dataObject.m.append(self.data[rownumber,:])
                elif ('cols' in selection) or ('columns' in selection):
                    if 'cols' in selection:
                        key = 'cols'
                    else:
                        key = columns
                    dataObject.x = None
                    dataObject.y = None
                    dataObject.m = None
                    if 'x' in selection[key]:
                        for rownumber in selection[key]['x']:
                            if rownumber is None:continue
                            if dataObject.x is None:dataObject.x = []
                            dataObject.x.append(self.data[:,rownumber])
                
                    if 'y' in selection[key]:
                        for rownumber in selection[key]['y']:
                            if rownumber is None:continue
                            if dataObject.y is None:dataObject.y = []
                            dataObject.y.append(self.data[:,rownumber])

                    if 'm' in selection[key]:
                        for rownumber in selection[key]['m']:
                            if rownumber is None:continue
                            if dataObject.m is None:dataObject.m = []
                            dataObject.m.append(self.data[:,rownumber])
                if dataObject.x is None:
                    if 'Channel0' in dataObject.info:
                        ch0 = int(output.info['Channel0'])
                    else:
                        ch0 = 0
                    dataObject.x = [numpy.arange(ch0,
                                 ch0 + len(dataObject.y[0])).astype(numpy.float)]
                if not ("selectiontype" in dataObject.info):
                    dataObject.info["selectiontype"] = "%dD" % len(dataObject.y) 
                return dataObject

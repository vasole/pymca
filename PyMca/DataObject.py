#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
import Numeric
import copy

class DataObject:
    def __init__(self):
        self.info = {}
        self.data = Numeric.array([])

    def getInfo(self):
        return self.info
    
    def getData(self):
        return self.data 

    if 0:
        def select(self,selection=None):
            if selection is None:
                return copy.deepcopy(self.data)
            else:
                print "Not implemented (yet)"
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
                if selection.has_key('rows'):
                    dataObject.x = None
                    dataObject.y = None
                    dataObject.m = None
                    if selection['rows'].has_key('x'):
                        for rownumber in selection['rows']['x']:
                            if rownumber is None:continue
                            if dataObject.x is None:dataObject.x = []
                            dataObject.x.append(self.data[rownumber,:])
                
                    if selection['rows'].has_key('y'):
                        for rownumber in selection['rows']['y']:
                            if rownumber is None:continue
                            if dataObject.y is None:dataObject.y = []
                            dataObject.y.append(self.data[rownumber,:])

                    if selection['rows'].has_key('m'):
                        for rownumber in selection['rows']['m']:
                            if rownumber is None:continue
                            if dataObject.m is None:dataObject.m = []
                            dataObject.m.append(self.data[rownumber,:])
                elif selection.has_key('cols') or selection.has_key('columns'):
                    if selection.has_key('cols'):key = 'cols'
                    else:key = columns
                    dataObject.x = None
                    dataObject.y = None
                    dataObject.m = None
                    if selection[key].has_key('x'):
                        for rownumber in selection[key]['x']:
                            if rownumber is None:continue
                            if dataObject.x is None:dataObject.x = []
                            dataObject.x.append(self.data[:,rownumber])
                
                    if selection[key].has_key('y'):
                        for rownumber in selection[key]['y']:
                            if rownumber is None:continue
                            if dataObject.y is None:dataObject.y = []
                            dataObject.y.append(self.data[:,rownumber])

                    if selection[key].has_key('m'):
                        for rownumber in selection[key]['m']:
                            if rownumber is None:continue
                            if dataObject.m is None:dataObject.m = []
                            dataObject.m.append(self.data[:,rownumber])
                if dataObject.x is None:
                    if dataObject.info.has_key('Channel0'):
                        ch0 = int(output.info['Channel0'])
                    else:
                        ch0 = 0
                    dataObject.x = [Numeric.arange(ch0,
                                 ch0 + len(dataObject.y[0])).astype(Numeric.Float)]
                if not dataObject.info.has_key("selectiontype"):
                    dataObject.info["selectiontype"] = "%dD" % len(dataObject.y) 
                return dataObject

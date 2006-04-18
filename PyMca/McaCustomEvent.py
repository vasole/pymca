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
try:
    from PyQt4 import QtCore
    QT4 = True
except:    
    import qt
    QT4 = False

if QT4:
    MCAEVENT = QtCore.QEvent.User
    #MCAEVENT = 12345

    class McaCustomEvent(QtCore.QEvent):
        def __init__(self,dict={}):
            self.dict = dict
            QtCore.QEvent.__init__(self,MCAEVENT)

        def type(self):
            print "called"
            return MCAEVENT
else:
    #MCAEVENT = qt.QUserEvent + 1
    MCAEVENT = 12345

    class McaCustomEvent(qt.QCustomEvent):
        def __init__(self,dict={}):
            qt.QCustomEvent.__init__(self,MCAEVENT)
            self.dict = dict
        
        
        
